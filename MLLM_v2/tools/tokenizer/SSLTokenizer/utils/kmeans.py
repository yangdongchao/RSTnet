
import torch
import torch.nn as nn


from .quantize import ResidualVQ
from .vocos import VocosBackbone
from .transformers import TransformerEncoder

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

class RepCodec(nn.Module):
    def __init__(
        self,
        codebook_size=8192,
        hidden_size=1024,
        codebook_dim=8,
        vocos_dim=384,
        vocos_intermediate_dim=2048,
        vocos_num_layers=12,
        num_quantizers=1,
        use_timbre_encoder=False,
        cfg=None,
    ):
        super().__init__()
        codebook_size = (
            cfg.codebook_size
            if cfg is not None and hasattr(cfg, "codebook_size")
            else codebook_size
        )
        codebook_dim = (
            cfg.codebook_dim
            if cfg is not None and hasattr(cfg, "codebook_dim")
            else codebook_dim
        )
        hidden_size = (
            cfg.hidden_size
            if cfg is not None and hasattr(cfg, "hidden_size")
            else hidden_size
        )
        vocos_dim = (
            cfg.vocos_dim
            if cfg is not None and hasattr(cfg, "vocos_dim")
            else vocos_dim
        )
        vocos_intermediate_dim = (
            cfg.vocos_intermediate_dim
            if cfg is not None and hasattr(cfg, "vocos_dim")
            else vocos_intermediate_dim
        )
        vocos_num_layers = (
            cfg.vocos_num_layers
            if cfg is not None and hasattr(cfg, "vocos_dim")
            else vocos_num_layers
        )
        num_quantizers = (
            cfg.num_quantizers
            if cfg is not None and hasattr(cfg, "num_quantizers")
            else num_quantizers
        )
        use_timbre_encoder = (
            cfg.use_timbre_encoder
            if cfg is not None and hasattr(cfg, "use_timbre_encoder")
            else use_timbre_encoder
        )

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.hidden_size = hidden_size
        self.vocos_dim = vocos_dim
        self.vocos_intermediate_dim = vocos_intermediate_dim
        self.vocos_num_layers = vocos_num_layers
        self.num_quantizers = num_quantizers
        self.use_timbre_encoder = use_timbre_encoder

        self.encoder = nn.Sequential(
            VocosBackbone(
                input_channels=self.hidden_size,
                dim=384,
                intermediate_dim=2048,
                num_layers=12,
                adanorm_num_embeddings=None
            ),
            nn.Linear(384, self.hidden_size)
        )
        self.decoder = nn.Sequential(
            VocosBackbone(
                input_channels=self.hidden_size,
                dim=384,
                intermediate_dim=2048,
                num_layers=12,
                adanorm_num_embeddings=None
            ),
            nn.Linear(384, self.hidden_size)
        )

        self.quantizer = ResidualVQ(
            input_dim=hidden_size,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_type="fvq",
            quantizer_dropout=0.0,
            commitment=0.15,
            codebook_loss_weight=1.0,
            use_l2_normlize=True,
        )

        if self.use_timbre_encoder:   #TODO: write encoder hidden (256) as a hyparam
            self.timbre_in = nn.Linear(hidden_size, 256)
            self.timbre_encoder = TransformerEncoder(
                enc_emb_tokens=None,
                encoder_layer=4,
                encoder_hidden=256,
                encoder_head=4,
                conv_filter_size=1024,
                conv_kernel_size=5,
                encoder_dropout=0.1,
                use_pe=False,
                cfg=None,
            )
            self.timbre_out = nn.Linear(256, hidden_size)
            self.timbre_linear = nn.Linear(hidden_size, hidden_size * 2)
            self.timbre_linear.bias.data[:hidden_size] = 1
            self.timbre_linear.bias.data[hidden_size:] = 0
            self.timbre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
            self.enc_ln = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.reset_parameters()

    def forward(self, x):

        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)

        if self.use_timbre_encoder:
            x_timbre = x
            x = x.transpose(1, 2)
            x = self.enc_ln(x)
            x = x.transpose(1, 2)

        (
            quantized_out,
            all_indices,
            all_commit_losses,
            all_codebook_losses,
            _,
        ) = self.quantizer(x)

        if self.use_timbre_encoder:
            x_timbre = x_timbre.transpose(1, 2)
            x_timbre = self.timbre_in(x_timbre)
            x_timbre = self.timbre_encoder(x_timbre, None, None)
            x_timbre = self.timbre_out(x_timbre)
            x_timbre = x_timbre.transpose(1, 2)
            spk_embs = torch.mean(x_timbre, dim=2)

            style = self.timbre_linear(spk_embs).unsqueeze(2)  # (B, 2d, 1)
            gamma, beta = style.chunk(2, 1)  # (B, d, 1)
            quantized_out = quantized_out.transpose(1, 2)
            quantized_out = self.timbre_norm(quantized_out)
            quantized_out = quantized_out.transpose(1, 2)
            quantized_out = quantized_out * gamma + beta
        

        x_rec = self.decoder(quantized_out)

        codebook_loss = (all_codebook_losses + all_commit_losses).mean()
        all_indices = all_indices

        return x_rec, codebook_loss, all_indices

    def quantize(self, x):
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)

        if self.use_timbre_encoder:
            x = x.transpose(1, 2)
            x = self.enc_ln(x)
            x = x.transpose(1, 2)

        (
            quantized_out,
            all_indices,
            all_commit_losses,
            all_codebook_losses,
            _,
        ) = self.quantizer(x)
        if all_indices.shape[0] == 1:
            return all_indices.squeeze(0), quantized_out.transpose(1, 2)
        return all_indices, quantized_out.transpose(1, 2)

    def reset_parameters(self):
        self.apply(init_weights)