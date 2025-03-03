import math
import torch
from torch import nn
import torch.onnx.operators
import torch.nn.functional as F
from timm.models.layers import Mlp


# sinusoidal positional encoding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :] * 1.0
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()

        self.dropout = dropout
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return F.dropout(x, self.dropout, training=self.training)


# style adaptive layer normalization
class AffineLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AffineLinear, self).__init__()
        affine = nn.Linear(in_dim, out_dim)
        self.affine = affine

    def forward(self, input):
        return self.affine(input)


class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.in_channel = normalized_shape
        self.norm = nn.LayerNorm(
            self.in_channel, elementwise_affine=False, eps=eps
        )  # y = (x - Ex) / sqrt(Var(x))

        self.style = AffineLinear(self.in_channel, self.in_channel * 2)
        self.style.affine.bias.data[: self.in_channel] = 1
        self.style.affine.bias.data[self.in_channel :] = 0

    def forward(self, input, condition):

        style = self.style(condition).unsqueeze(1)  # (B, 1, 2C)

        gamma, beta = style.chunk(2, dim=-1)  # (B, 1, C), (B, 1, C)

        out = self.norm(input)  # (B, T, C)

        out = gamma * out + beta
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size=1024,
        num_heads=16,
        dropout=0.1,
        ffn_dropout=0.1,
        attention_dropout=0.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=attention_dropout,
            bias=True,
            batch_first=True,
        )

        self.diffusion_mlp = nn.Linear(self.hidden_size, self.hidden_size)
        self.cond_mlp = nn.Linear(self.hidden_size, self.hidden_size)

        self.layer_norm1 = StyleAdaptiveLayerNorm(self.hidden_size)
        self.layer_norm2 = StyleAdaptiveLayerNorm(self.hidden_size)

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size * 4,
            act_layer=approx_gelu,
            drop=ffn_dropout,
        )

    def forward(self, x, diffusion_step, key_padding_mask=None):
        residual = x
        x = self.layer_norm1(x, diffusion_step)
        x, _ = self.self_attn(
            query=x, key=x, value=x, key_padding_mask=key_padding_mask
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.layer_norm2(x, diffusion_step)
        x = self.mlp(x)
        x = residual + x

        return x


class DiffTransformer(nn.Module):
    def __init__(
        self,
        hidden_size=1024,
        num_heads=16,
        num_layers=16,
        dropout=0.1,
        ffn_dropout=0.1,
        attention_dropout=0.0,
    ):
        super().__init__()

        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    ffn_dropout=ffn_dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.diff_step_embedding = SinusoidalPosEmb(hidden_size)
        self.diff_step_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.position_embedding = PositionalEncoding(hidden_size, dropout=0.0)

        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.reset_parameters()

    def forward(self, x, diffusion_step, cond, x_mask):
        # x: (B, T, C)
        # cond: (B, T, C)
        # x_mask: (B, T) mask for padding, mask is 0 for padding

        # condtion mlp
        cond = self.cond_mlp(cond)  # (B, T, C)

        # diffusion step embedding
        diffusion_step = self.diff_step_embedding(diffusion_step).to(x.device)
        diffusion_step = self.diff_step_mlp(diffusion_step)  # (B, C)

        # positional embedding
        pos_emb = self.position_embedding(x)

        # add positional embedding and condition
        x = x + pos_emb
        x = x + cond

        # pay attention, x_mask is 0 for padding, so we need to invert it for key_padding_mask
        key_padding_mask = ~x_mask.bool()

        for layer in self.transformer_layers:
            x = layer(x, diffusion_step, key_padding_mask=key_padding_mask)

        return x

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.MultiheadAttention):
                if m._qkv_same_embed_dim:
                    nn.init.normal_(m.in_proj_weight, std=0.02)
                else:
                    nn.init.normal_(m.q_proj_weight, std=0.02)
                    nn.init.normal_(m.k_proj_weight, std=0.02)
                    nn.init.normal_(m.v_proj_weight, std=0.02)

                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.0)
                    nn.init.constant_(m.out_proj.bias, 0.0)
                if m.bias_k is not None:
                    nn.init.xavier_normal_(m.bias_k)
                if m.bias_v is not None:
                    nn.init.xavier_normal_(m.bias_v)

            elif (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
            ):
                m.weight.data.normal_(0.0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

        self.apply(_reset_parameters)


# if __name__ == "__main__":
#     model = DiffTransformer(hidden_size=1024, num_heads=16, num_layers=16)
#     # count number of parameters
#     num_params = sum(p.numel() for p in model.parameters())
#     print(f"Number of parameters: {num_params/1e6:.2f}M")
#     x = torch.randn(2, 100, 1024)
#     diffusion_step = torch.randint(0, 100, (2,))
#     cond = torch.randn(2, 100, 1024)
#     x_mask = torch.randint(0, 2, (2, 100))
#     out = model(x, diffusion_step, cond, x_mask)
#     print(out.shape)
