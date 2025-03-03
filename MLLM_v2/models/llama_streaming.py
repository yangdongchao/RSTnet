# updated by Dongchao
# we focus on combine llama and local transformer
# Derived from https://github.com/microsoft/LoRA and Lightning AI. 
#  ------------------------------------------------------------------------------------------

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self
from models import lit_model
from models import mlp
from models.lit_model import GPT as BaseModel
from models.lit_model import Block as BaseBlock
from models.lit_model import CausalSelfAttention as BaseCausalSelfAttention
from models.lit_model import KVCache, apply_rope, RingKVCache, KVCacheResult
from functools import partial
from modules.streaming import StreamingModule, StreamingContainer 
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Mapping, Optional, TypeVar, Union
from models.config import Config as BaseConfig
from modules.transformer import StreamingTransformer

def map_old_state_dict_weights(state_dict: Dict, mapping: Mapping, prefix: str) -> Dict:
    for checkpoint_name, attribute_name in mapping.items():
        full_checkpoint_name = prefix + checkpoint_name
        if full_checkpoint_name in state_dict:
            full_attribute_name = prefix + attribute_name
            state_dict[full_attribute_name] = state_dict.pop(full_checkpoint_name)
    return state_dict


class LoRALayer(nn.Module):
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float):
        """Store LoRA specific attributes in a class.

        Args:
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        """
        super().__init__()
        assert r >= 0
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False


class LoRALinear(LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs: Any,
    ):
        """LoRA wrapper around linear class.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.linear.weight`
            2. LoRA A matrix as `self.lora_A`
            3. LoRA B matrix as `self.lora_B`
        Only LoRA's A and B matrices are updated, pretrained weights stay frozen.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        """
        super().__init__(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(torch.empty((r, in_features)))
            self.lora_B = nn.Parameter(torch.empty((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset all the weights, even including pretrained ones."""
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def get_lora_AB(self) -> torch.Tensor:
        """Return merged lora_A and lora_B matrices with the same shape as the pretrained weights."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def merge(self) -> None:
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""
        if self.r > 0 and not self.merged:
            pretrained_dtype = self.linear.weight.data.dtype
            lora_data = self.get_lora_AB()
            # if only the pretrained are in quantized form - dequantize, sum with LoRA and quantize the result
            if pretrained_dtype == torch.uint8:
                import bitsandbytes as bnb

                weight = self.linear.weight
                # dequantize the pretrained weights
                weight_data = bnb.functional.dequantize_4bit(weight.data, weight.quant_state).to(lora_data.dtype)
                # add pretrained and LoRA weights
                weight_data += lora_data
                # assign updated weights and quantize by moving to CUDA device
                self.linear.weight = bnb.nn.Params4bit(weight_data, requires_grad=False, **weight.__dict__)
                self.linear.weight.cuda(weight.device)
            else:
                # self.linear might be on CPU and lora_data on CUDA
                # the inplace add will preserve the dtype of linear.weight
                self.linear.weight.data += lora_data.to(device=self.linear.weight.data.device)
            self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if weights are merged or rank is less or equal to zero (LoRA is disabled) - it's only a regular nn.Linear forward pass;
        # otherwise in addition do the forward pass with LoRA weights and add it's output to the output from pretrained weights
        pretrained = self.linear(x)
        if self.r == 0 or self.merged:
            return pretrained
        lora = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return pretrained + lora


class LoRAQKVLinear(LoRALinear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        head_size: int,
        n_head: int,
        n_query_groups: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: Union[bool, Tuple[bool, bool, bool]] = False,
        **kwargs: Any,
    ):
        """LoRA wrapper around linear class that is used for calculation of q, k and v matrices.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.linear.weight`
            2. LoRA A matrix as `self.lora_A`
            3. LoRA B matrix as `self.lora_B`
        Only LoRA's A and B matrices are updated, pretrained weights stay frozen.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            head_size: size of a single attention head
            n_head: number of attention heads
            n_query_groups: number of query groups (see diagram in `litgpt/config.py`)
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
            enable_lora: MergeLinear class is for attention mechanism where qkv are calculated with a single weight matrix. If we
                don't want to apply LoRA we can set it as False. For example if we want to apply LoRA only to `query`
                and `value` but keep `key` without weight updates we should pass `[True, False, True]`
        """
        super(LoRALinear, self).__init__(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)
        self.head_size = head_size
        self.n_head = n_head
        self.n_query_groups = n_query_groups
        if isinstance(enable_lora, bool):
            enable_lora = [enable_lora] * 3
        assert len(enable_lora) == 3
        self.enable_lora = enable_lora

        # Actual trainable parameters
        # To better understand initialization let's imagine that we have such parameters:
        # ⚬ in_features: 128 (embeddings_size)
        # ⚬ out_features: 384 (3 * embedding_size)
        # ⚬ r: 2
        # ⚬ enable_lora: [True, False, True]
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(torch.empty((r * sum(enable_lora), in_features)))  # (4, 128)
            enable_q, enable_k, enable_v = enable_lora
            # qkv_shapes will be used to split a tensor with weights correctly
            qkv_shapes = (
                # if `head_size` is explicitly specified in the config, `n_embd` (or `in_features`)
                # might not be equal to `head_size * n_head`, thus we use it directly here
                head_size * n_head * enable_q,
                head_size * n_query_groups * enable_k,
                head_size * n_query_groups * enable_v,
            )
            self.qkv_shapes = [s for s in qkv_shapes if s]
            self.lora_B = nn.Parameter(torch.empty(sum(self.qkv_shapes), r))  # (256, 2))
            # Notes about shapes above
            # - self.lora_A has shape (4, 128): 4 because rank is 2 and LoRA is applied only to two matrices;
            # 128 is the input size of the x (embedding size). (4, 128) and not (128, 4) because later on in
            # F.linear function weights are automatically transposed. In addition conv1d requires channels to
            # be before seq length
            # - self.lora_B has shape (256, 2): 256 because LoRA is applied only to two matrices, so the output is
            # 128*2; 2 tells to have two channels per group for group convolution

            # Scaling:
            # This balances the pretrained model`s knowledge and the new task-specific adaptation
            # https://lightning.ai/pages/community/tutorial/lora-llm/
            # So, set alpha to 1.0 to fully add LoRA. If the LoRA seems to have too much effect (i.e., overfitted), set
            # alpha to lower value. If the LoRA seems to have too little effect, set alpha to higher than 1.0. You can
            # tune these values to your needs. This value can be even slightly greater than 1.0!
            # https://github.com/cloneofsimo/lora
            self.scaling = self.lora_alpha / self.r

            self.reset_parameters()

    @property
    def lora_ind(self) -> torch.Tensor:
        """Lazy creation of a buffer with LoRA indices to overcome the limitation when FSDP with meta device is used."""
        # Indices are needed to properly pad weight updates with zeros.
        if not hasattr(self, "_lora_ind"):
            enable_q, enable_k, enable_v = self.enable_lora
            qkv_group_size = self.n_head // self.n_query_groups + 2
            candidate_indices = range(self.linear.out_features)
            lora_ind = []
            if enable_q:
                q_ind = [x for x in candidate_indices if (x // self.head_size) % qkv_group_size < qkv_group_size - 2]
                lora_ind.extend(q_ind)
            if enable_k:
                k_ind = [x for x in candidate_indices if (x // self.head_size) % qkv_group_size == qkv_group_size - 2]
                lora_ind.extend(k_ind)
            if enable_v:
                v_ind = [x for x in candidate_indices if (x // self.head_size) % qkv_group_size == qkv_group_size - 1]
                lora_ind.extend(v_ind)
            self.register_buffer(
                "_lora_ind", torch.tensor(lora_ind, device=self.linear.weight.device), persistent=False
            )

        return self._lora_ind

    def zero_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Properly pad the last dimension of weight updates with zeros.

        If, based on `self.enable_lora`, we want to fine-tune queries and values, but not keys,
        then the weights update should be:

        [[ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,],
         [....................................],
         [ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,]]
            ↑              ↑            ↑
        ________________________________________
        | query         | key       | value    |
        ----------------------------------------
        For Llama2's GQA support, Q, K, and V weights are interleaved, so that weights for grouped
        queries are adjacent to their associated key and value weights.
        For example, suppose we have n_head = 12 with 3 query groups.
        Then along the embedding dimension the interleaved weights would look like

        [Q, Q, Q, Q, K, V, Q, Q, Q, Q, K, V, Q, Q, Q, Q, K, V],

        where each Q, K, and V has size head_size.

        In this case, the previously-described weight update applies separately to each
        individual block, so the update will take the form

        [[ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW, ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW, ...],
         [.............................................................................],
         [ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW, ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW, ...]]
             ↑              ↑            ↑        ↑             ↑            ↑
        ________________________________________________________________________________
        | q block 1 | k block 1  | v block 1 | q block 2 |  k block 2 |  v block 2 | ...
        --------------------------------------------------------------------------------
        Note that in the above diagram, the size of each q block will equal q_per_kv
        times the size of each k and v block.

        Args:
            x: tensor with weights update that will be padded with zeros if necessary

        Returns:
            A tensor with weight updates and zeros for deselected q, k or v
        """
        # we need to do zero padding only if LoRA is disabled for one of QKV matrices
        if all(self.enable_lora):
            return x

        # Let's image that:
        # ⚬ input x has shape (64, 64, 256): (batch_size, sequence_length, embeddings_size)
        # ⚬ embeddings_size: 128
        # ⚬ self.linear.out_features: 384 (3 * embeddings_size)
        # ⚬ enable_lora: [True, False, True]
        # Then x has embeddings_size of 256 (2 * 128 as enable_lora only for query and value, not keys) and expected
        # embeddings_size is 384 (self.linear.out_features), so that means that we need to pad from 256 to 384 with zeros, but
        # only for key updates (this is where self.lora_ind comes in handy)

        result = x.new_zeros(*x.shape[:-1], self.linear.out_features)  # (64, 64, 384)
        if result.device.type == "mps":
            result[..., self.lora_ind] = x
            return result
        else:
            return result.index_copy_(dim=-1, index=self.lora_ind, source=x)  # (64, 64, 384)

    def conv1d(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """An extension of the `torch.nn.functional.conv1d` function with a logic specific to grouped queries.

        If the number of heads is equal to the number of query groups - grouped queries are disabled
        (see scheme in `litgpt/config.py:Config`). In this case the combined QKV matrix consists of equally sized
        query, key and value parts, which means we can utilize `groups` argument from `conv1d`: with this argument the
        input and weight matrices will be splitted in equally sized parts and applied separately (like having multiple
        conv layers side by side).

        Otherwise QKV matrix consists of unequally sized parts and thus we have to split input and weight matrices manually,
        apply each part of the weight matrix to the corresponding input's part and concatenate the result.

        Args:
            input: input matrix of shape (B, C, T)
            weight: weight matrix of shape (C_output, rank, 1).
                "C_output" is defined as a sum of embedding sizes for each enabled LoRA layer (see init method of the class).

        Returns:
            A tensor with a shape (B, C_output, T)

        """
        if self.n_head == self.n_query_groups:
            return F.conv1d(input, weight, groups=sum(self.enable_lora))  # (B, C_output, T)

        # Notation:
        # ⚬ N: number of enabled LoRA layers (self.enable_lora)
        # ⚬ C_output': embeddings size for each LoRA layer (not equal in size)
        # ⚬ r: rank of all LoRA layers (equal in size)

        input_splitted = input.chunk(sum(self.enable_lora), dim=1)  # N * (B, C // N, T)
        weight_splitted = weight.split(self.qkv_shapes)  # N * (C_output', r, 1)
        return torch.cat(
            [F.conv1d(a, b) for a, b in zip(input_splitted, weight_splitted)],
            dim=1,  # (B, C_output', T)
        )  # (B, C_output, T)

    def get_lora_AB(self) -> torch.Tensor:
        """Return merged lora_A and lora_B matrices with the same shape as the pretrained weights."""
        # Let's assume that:
        # ⚬ self.linear.weight.data: (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data: (4, 128)
        # ⚬ self.lora_B.data: (256, 2)
        lora = self.conv1d(
            self.lora_A.data.unsqueeze(0),  # (4, 128) -> (1, 4, 128)
            self.lora_B.data.unsqueeze(-1),  # (256, 2) -> (256, 2, 1)
        ).squeeze(0)  # (1, 4, 128) @ (256, 2, 1) -> (1, 256, 128) -> (256, 128)
        return self.zero_pad(lora.T * self.scaling).T  # (256, 128) after zero_pad (384, 128)

    def merge(self) -> None:
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""
        if self.r > 0 and any(self.enable_lora) and not self.merged:
            super().merge()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass.

        If LoRA's weights are merged with pretrained ones then it's a simple matrix multiplication.
        If not, then multiply pretrained weights with input, apply LoRA on input and do summation.

        Args:
            x: input tensor of shape (batch_size, context_length, embedding_size)

        Returns:
            Output tensor of shape (batch_size, context_length, 3 * embedding_size)
        """

        # Let's assume that:
        # ⚬ x: (64, 64, 128) or (batch_size, context_length, embedding_size)
        # ⚬ self.linear.weight: (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data: (4, 128)
        # ⚬ self.lora_B.data: (256, 2)

        # if weights are merged or LoRA is disabled (r <= 0 or all `enable_lora` are False) - it's only a regular nn.Linear forward pass;
        # otherwise in addition do the forward pass with LoRA weights and add it's output to the output from pretrained weights
        pretrained = self.linear(x)
        if self.r == 0 or not any(self.enable_lora) or self.merged:
            return pretrained
        after_A = F.linear(self.lora_dropout(x), self.lora_A)  # (64, 64, 128) @ (4, 128) -> (64, 64, 4)
        # For F.conv1d:
        # ⚬ input: input tensor of shape (mini-batch, in_channels, iW)
        # ⚬ weight: filters of shape (out_channels, in_channels/groups, kW)
        after_B = self.conv1d(
            after_A.transpose(-2, -1),  # (64, 64, 4) -> (64, 4, 64)
            self.lora_B.unsqueeze(-1),  # (256, 2) -> (256, 2, 1)
        ).transpose(-2, -1)  # (64, 4, 64) @ (256, 2, 1) -> (64, 256, 64) -> (64, 64, 256)
        lora = self.zero_pad(after_B) * self.scaling  # (64, 64, 256) after zero_pad (64, 64, 384)
        return pretrained + lora


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """Freeze all modules except LoRA's and depending on 'bias' value unfreezes bias weights.

    Args:
        model: model with LoRA layers
        bias:
            ``"none"``: all bias weights will be frozen,
            ``"lora_only"``: only bias weight for LoRA layers will be unfrozen,
            ``"all"``: all bias weights will be unfrozen.

    Raises:
        NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
    """
    # freeze all layers except LoRA's
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False

    # depending on the `bias` value unfreeze bias weights
    if bias == "none":
        return
    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_filter(key: str, value: Any) -> bool:
    return "lora_" in key


@dataclass
class Config(BaseConfig):
    """
    Args:
        lora_r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
            the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
        lora_alpha: alpha is needed for scaling updates as alpha/r
            "This scaling helps to reduce the need to retune hyperparameters when we vary r"
            https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
        lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        lora_*: whether to apply LoRA to the specified weights or not
    """

    # lora
    lora_r: int = 0
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    lora_query: bool = False
    lora_key: bool = False
    lora_value: bool = False
    lora_projection: bool = False
    lora_mlp: bool = False
    lora_head: bool = False

    # local transformer config
    audio_card: int = 2048
    codecformer_dim: int = 1024
    n_q: int = 9
    dep_q: int = 8
    codecformer_heads: int = 32
    codecformer_layers: int = 6
    codecformer_hidden_scale: float = 4.5
    causal: bool = True
    codecformer_multi_linear: bool = True
    codecformer_weights_per_step: bool = True
    codecformer_dim_feedforward: int = 1024
    codecfomer_norm: str = "rms_norm_f32"
    codecformer_bias_proj: bool = False
    codecfomer_norm_emb: bool = False
    context: int =  3000


    @property
    def mlp_class(self) -> Type:
        model_class = globals().get(self.mlp_class_name, None)
        return model_class

class ScaledEmbedding(nn.Embedding):
    """Boost learning rate for embeddings (with `scale`).

    Args:
        norm (bool): if True, uses a layer norm after the embedding.
        zero_idx (int): special value indicating that the output should be exactly 0.
    """

    def __init__(self, *args, norm: bool = False, zero_idx: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = None
        if norm:
            self.norm = create_norm_fn("layer_norm", self.embedding_dim)
        assert zero_idx < 0, "Please use negative values for the zero_idx."
        self.zero_idx = zero_idx

    def forward(self, input, *args, **kwargs):
        is_zero = input == self.zero_idx
        zero = torch.zeros(1, dtype=input.dtype, device=input.device)
        input = input.clamp(min=0)
        y = super().forward(input, *args, **kwargs)
        if self.norm is not None:
            y = self.norm(y)
        y = torch.where(is_zero[..., None], zero, y)
        return y


class GPT(BaseModel, StreamingContainer):
    ''' The audio-text LLM (MLLM), supporting streaming inference
    '''
    def __init__(self, config: Config):
        super().__init__(config)
        assert config.padded_vocab_size is not None
        self.config = config
        #------------------ global transformer config --------------
        self.lm_head = LoRALinear(
            config.n_embd,
            config.padded_vocab_size,
            bias=config.lm_head_bias,
            r=(config.lora_r if config.lora_head else 0),
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
        #self.delays = [0, 0, 1, 1, 1, 1, 1, 1, 1] # we should change it into configureable
        self.dep_q = config.dep_q
        self.transformer = LLAMAStreamingTransformer(config) # init the global transformer
        self.max_seq_length = self.config.block_size

        #------------------ local transformer --------------
        EmbeddingFactory = partial(
            ScaledEmbedding,
            norm=config.codecfomer_norm_emb,
            zero_idx=self.zero_token_id,
        )
        self.input_emb = nn.ModuleList([EmbeddingFactory(config.audio_card + 1, config.n_embd) for _ in range(config.n_q)]) # build embedding mapping for each codebook config.dim 
        kwargs_codec = {}
        kwargs_codec["context"] = None
        kwargs_codec['gating'] = 'silu'
        kwargs_codec['positional_embedding'] = 'none' # we donot use the positional embedding for the local transformer
        kwargs_codec['layer_scale'] = None
        kwargs_codec['causal'] = True
        kwargs_codec['max_period'] = 10000
        kwargs_codec['gating'] = 'silu'

        if config.codecformer_weights_per_step:
            kwargs_codec["weights_per_step"] = config.dep_q
        if config.codecformer_multi_linear: #
            # One linear layer per codebook to project different informations from the main model.
            self.codecformer_in = nn.ModuleList(
                [nn.Linear(config.n_embd, config.codecformer_dim, bias=False) for _ in range(config.dep_q)]
            ) # for the depth transformer, we use different linear layer to map features dep_q=8. We only output our stream
        else:
            self.codecformer_in = nn.ModuleList(
                [nn.Linear(config.n_embd, config.codecformer_dim, bias=False)]
            )
        # Only using up to dep_q - 1 because the last codebook is never an input to Depformer.
        self.codecformer_emb = nn.ModuleList(
            [EmbeddingFactory(config.audio_card + 1, config.codecformer_dim) for _ in range(config.dep_q - 1)]
        )
        self.codecformer_text_emb = EmbeddingFactory(config.padded_vocab_size, config.codecformer_dim) # init a new table for local transformer
        self.codecformer = StreamingTransformer(
            d_model=config.codecformer_dim,
            dim_feedforward=config.codecformer_dim_feedforward,
            norm=config.codecfomer_norm,
            num_heads=config.codecformer_heads,
            num_layers=config.codecformer_layers,
            **kwargs_codec,
        )
        self.codecformer.set_streaming_propagate(False) # it will not set as streaming
        self.audio_linears = nn.ModuleList(
            [nn.Linear(config.codecformer_dim, config.audio_card, bias=config.codecformer_bias_proj) for _ in range(config.dep_q)]
        ) # output for each layer
    
    def print_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params_m = total_params / 1_000_000  # 转换为百万
        print(f"Total trainable parameters: {total_params_m:.2f} M")

    @property
    def zero_token_id(self) -> int:
        """Special value in the input tokens, indicating that no sampling should
        happen for that value, and no input should be given to the model."""
        return -1
    
    @property
    def text_initial_token_id(self) -> int:
        """Token id for the start of sequence (text)."""
        #"128002": {"content": "<|reserved_special_token_0|>",
        return 128010  # llama3 tokenizer: reserved tokens from 128002-128255

    @property
    def initial_token_id(self) -> int:
        """Token id for the start of sequence (audio)."""
        return self.config.audio_card
    
    @property
    def num_codebooks(self) -> int:
        return self.config.n_q + 1

    @property
    def num_audio_codebooks(self) -> int:
        return self.config.n_q
    
    @property
    def audio_offset(self) -> int:
        return 1
    
    @property
    def ungenerated_token_id(self) -> int:
        """Special value that can be provided in the prompt to indicate that this specific
        value should be predicted and sampled. This allows for partial teacher forcing, by generating
        one modality, with the other one fixed.
        """
        return -2

    @property
    def device(self):
        first_param = next(iter(self.parameters()))
        return first_param.device

    def _get_initial_token(self) -> torch.Tensor:
        # Returns the initial token that will be fed to the model to predict the very first timestep. 
        #init one frame (1,17,1) post-training (multi-streaming)or [1, 9, 1] pre-training
        # The output shape will be [B, K, 1].
        device = next(iter(self.parameters())).device
        zero = torch.full([1, 1, 1], self.zero_token_id, device=device, dtype=torch.long)
        special = torch.full_like(zero, self.initial_token_id)

        text_special = torch.full_like(zero, self.text_initial_token_id)
        audio_token = special
        text_token = text_special
        audio_token = audio_token.expand(-1, self.num_audio_codebooks, -1)
        token = torch.cat([text_token, audio_token], dim=1)
        return token

    def forward(self, sequence: torch.Tensor, input_pos: Optional[torch.Tensor] = None, lm_head_chunk_size: int = 0):
        B, K, S = sequence.shape  # 
        global_start_frame = self._get_initial_token() # get the 
        global_start_frame = global_start_frame.repeat(B, 1, 1) # 
        global_input_sequence = torch.cat([global_start_frame, sequence[:,:,:-1]], dim=2) # add start token
        transformer_out, text_logits = self.forward_global(global_input_sequence) #
        text_logits = text_logits.squeeze(1) # B, T, D
        
        text_indices =  sequence[:,0,:]
        local_start_token = self.codecformer_text_emb(text_indices) # using text embedding for local start token, B,T,D
        local_sequence = sequence[:,1:self.config.dep_q+1,:] # the input for local sequence
        audio_logits = self.forward_local(local_start_token, local_sequence, transformer_out) # B, T, 8, card
        return audio_logits, text_logits
    
    def forward_global(self, sequence: torch.Tensor):
        ''' The global forward function. it should includes three types
            only text
            text + speech 
            speech may includes 1 streaming (pre-training) or 2 streaming (post-training)
            if for only text, we can add audio_empty token to denotes the empty for speech. Also, we introduce random PAD for text word
            if for only speech, we can add text_empty token 
        '''
        B, K, T = sequence.shape 
        assert (K == self.num_codebooks), f"Sequence shape {sequence.shape} must match the number of codebooks."
        if self.max_seq_length < T: # check the max length
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")
        cos = self.cos
        sin = self.sin
        input_sequence = sequence
        input_ = None
        for cb_index in range(self.num_audio_codebooks): # 16
            audio_emb = self.input_emb[cb_index](input_sequence[:, cb_index + self.audio_offset]) # different codebook use different layer
            input_ = audio_emb if input_ is None else input_ + audio_emb # using add operation to merge all of the information
        text_emb = self.transformer.wte(input_sequence[:,0])
        input_ = text_emb if input_ is None else input_ + text_emb # similarly, add the information
        if self.config.scale_embeddings: # currentlt, it is false
            input_ = input_ * (self.config.n_embd**0.5)
        transformer_out = self.transformer(input_, cos, sin) # this need to be check. I am not sure whether we should output norm feature for local 
        text_logits = self.lm_head(transformer_out)  # (b, t, vocab_size)
        return transformer_out, text_logits

    def forward_local(self, local_start_token: torch.Tensor, sequence: torch.Tensor, transformer_out: torch.Tensor) -> torch.Tensor:
        ''' the local_start_token already been the features
        '''
        B, K, S = sequence.shape
        assert (K == self.config.dep_q), f"Sequence shape {sequence.shape} must match the moshi stream output."
        depformer_input = transformer_out
        local_inputs = []
        local_start_token = local_start_token.reshape(-1, local_start_token.shape[-1]) # transfer to 
        
        local_inputs.append(local_start_token)
        different_view_depformer = []
        for cb_index in range(self.config.dep_q-1): # 7
            local_token_input = self.codecformer_emb[cb_index](sequence[:,cb_index:cb_index+1,:]) # get the local embedding, B, 1, T,D
            local_token_input = local_token_input.reshape(-1, local_token_input.shape[-1])
            local_inputs.append(local_token_input) # B*T,D

        for cb_index in range(self.config.dep_q):
            tmp_dep_input = self.codecformer_in[cb_index](depformer_input) # apply different view for different layer
            tmp_dep_input = tmp_dep_input.reshape(-1, tmp_dep_input.shape[-1])
            different_view_depformer.append((tmp_dep_input+local_inputs[cb_index]).unsqueeze(1)) #B*T,1, D

        real_depformer_input = torch.cat(different_view_depformer, dim=1) # B*T, 8, D
        # The streaming state of the depformer ensures that the proper layer is run.
        dep_output = self.codecformer(real_depformer_input) # B*T, 8, D
        logits = []
        for depformer_cb_index in range(self.config.dep_q):
            tmp_logit = self.audio_linears[depformer_cb_index](dep_output[:,depformer_cb_index:depformer_cb_index+1,:]) # B*T,1,card
            tmp_logit = tmp_logit.reshape(B, -1, 1, tmp_logit.shape[-1]) # B, T, 1, card
            logits.append(tmp_logit)
        logits = torch.cat(logits, dim=2)  # B, T, 8, card
        assert logits.dim() == 4, logits.shape  # ?
        return logits

    def forward_codecformer(self, codecformer_cb_index: int, sequence: torch.Tensor, transformer_out: torch.Tensor) -> torch.Tensor:
        ''' for inference only
        '''
        B, K, S = sequence.shape
        assert (K == 1), f"Codebooks for Depformer streaming should be passed 1 by 1, got {K}."
        assert (S == 1), f"Steps for Depformer streaming should be passed 1 by 1, got {S}."
        assert (transformer_out.shape[1] == 1), "Transformer out should be a for a single step."
        last_token_input: tp.Optional[torch.Tensor] = None
        depformer_input = transformer_out # 
        depformer_input = self.codecformer_in[codecformer_cb_index](depformer_input) # transfer the feature space
        if codecformer_cb_index == 0:
            last_token_input = self.codecformer_text_emb(sequence[:, 0]) # using text emb
        else:
            last_token_input = self.codecformer_emb[codecformer_cb_index - 1](sequence[:, 0])
        depformer_input = depformer_input + last_token_input # add them
        assert depformer_input.shape[1] == 1
        # depformer_input is [B, 1, depformer_dim].
        # The streaming state of the depformer ensures that the proper layer is run.
        dep_output = self.codecformer(depformer_input)
        logits = self.audio_linears[codecformer_cb_index](dep_output)
        logits = logits[:, None]
        assert logits.dim() == 4, logits.shape  # [B, Ka, S, card]
        return logits

    
    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`. Unused method left for completeness."""
        super()._init_weights(module)
        if isinstance(module, LoRALinear):
            module.reset_parameters()

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {"lm_head.weight": "lm_head.linear.weight", "lm_head.bias": "lm_head.linear.bias"}
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

@dataclass
class _TransformerState: # we use it to record the offset. This can tell us about the inference depth
    offset: torch.Tensor

    def reset(self):
        self.offset.zero_()

class LLAMAStreamingTransformer(StreamingModule[_TransformerState]):
    def __init__(self, config):
        # nn.Module.__init__(self)
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.padded_vocab_size, config.n_embd)
        self.h = nn.ModuleList(Block(config, block_idx) for block_idx in range(config.n_layer))
        self.ln_f = config.norm_class(config.n_embd, eps=config.norm_eps)
    
    def _init_streaming_state(self, batch_size: int) -> _TransformerState:
        device = next(self.parameters()).device
        return _TransformerState(offset=torch.zeros(1, device=device, dtype=torch.long))
    
    def forward(self, x: torch.Tensor, cos, sin):
        B, T, C = x.shape
        state = self._streaming_state
        if state is None:
            offset = torch.zeros(1, dtype=torch.long, device=x.device)
        else:
            offset = state.offset
        for block in self.h:
            x = block(x, cos, sin)
        x = self.ln_f(x)
        if state is not None:
            state.offset.add_(T)
        return x 
    

@dataclass
class _LayerState:
    offset_cpu: int

    def reset(self):
        self.offset_cpu = 0

class Block(BaseBlock, StreamingModule[_LayerState]):
    def __init__(self, config: Config, block_idx: int) -> None:
        #nn.Module.__init__(self)
        super().__init__(config, block_idx)
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses this configuration:"
                " non-parallel residual and shared attention norm."
            )
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, block_idx)
        self.post_attention_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps) if config.post_attention_norm else nn.Identity()
        )
        self.norm_2 = None if config.shared_attention_norm else config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)
        self.post_mlp_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps) if config.post_mlp_norm else nn.Identity()
        )
        self.config = config
    
    def _init_streaming_state(self, batch_size: int) -> _LayerState:
        return _LayerState(offset_cpu=0) # record the offset for data

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x_normed = self.norm_1(x)
        attention_output = self.attn(x_normed, cos, sin)
        attention_output = self.post_attention_norm(attention_output)
        if self.config.parallel_residual: # false
            x_normed = x_normed if self.config.shared_attention_norm else self.norm_2(x)
            x = self.mlp(x_normed) + attention_output + x
        else:
            x = attention_output + x
            x = self.post_mlp_norm(self.mlp(self.norm_2(x))) + x
        state = self._streaming_state
        # print('block state ', state)
        if state:
            state.offset_cpu += x.shape[1]
        return x


@dataclass
class _MHAState:
    kv_cache: RingKVCache
    offset: torch.Tensor
    offset_cpu: int

    def reset(self):
        self.kv_cache.reset() # rest
        self.offset.zero_()
        self.offset_cpu = 0
    
class CausalSelfAttention(BaseCausalSelfAttention, StreamingModule[_MHAState]):
    def __init__(self, config: Config, block_idx: int) -> None:
        # Skip the parent class __init__ altogether and replace it to avoid
        # useless allocations
        super().__init__(config, block_idx)
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = LoRAQKVLinear(
            in_features=config.n_embd,
            out_features=shape,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            enable_lora=(config.lora_query, config.lora_key, config.lora_value),
            bias=config.bias,
            # for MQA/GQA support
            head_size=config.head_size,
            n_head=config.n_head,
            n_query_groups=config.n_query_groups,
        )
        # output projection
        # if `head_size` is explicitly specified in the config, `n_emd` might not be equal to `head_size * n_head`
        self.proj = LoRALinear(
            config.head_size * config.n_head,
            config.n_embd,
            bias=config.bias,
            r=(config.lora_r if config.lora_projection else 0),
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
        # disabled by default
        self.kv_cache: Optional[KVCache] = None
        self.apply_sliding_window_attention = (
            config.sliding_window_size is not None and
            block_idx % config.sliding_window_layer_placing == 0
        )
        self.config = config

    def _init_streaming_state(self, batch_size: int) -> _MHAState:
        if self.config.context is None:
            if self.config.weights_per_step:
                capacity = self.config.weights_per_step
            else:
                raise RuntimeError(
                    "Cannot create a streaming KVCache without a context to estimate capacity."
                )
        else:
            capacity = self.config.context # we will use the 2k context
        #print('capacity ', capacity)
        device = self.proj.linear.weight.device
        # TODO: the following estimation will not work great with FSDP.
        dtype = self.proj.linear.weight.dtype
        kv_cache = RingKVCache(
            batch_size, self.config.n_head, self.config.head_size, capacity, device, dtype
        )
        return _MHAState(
            kv_cache,
            offset=torch.zeros(1, device=device, dtype=torch.long),
            offset_cpu=0,
        )

    def _complete_kv(self, k, v) -> KVCacheResult:
        state = self._streaming_state
        if state is None:
            return KVCacheResult.from_kv(k, v)
        else:
            return state.kv_cache.complete(k, v)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        ''' we re-write the forward function.
        '''
        state = self._streaming_state # it will call _init_streaming_state?
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        if state is None: # if we are not in the streaming, we set the offset as zero
            offset = torch.zeros(1, device=x.device, dtype=torch.long)
            offset_cpu = 0
        else:
            assert self.config.causal, "Streaming only available for causal"
            offset = state.offset
            offset_cpu = state.offset_cpu
        
        qkv = self.attn(x) # get q k v

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        if self.config.n_query_groups != self.config.n_head and (self.config.n_query_groups != 1):
            k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)
        if state is not None:
            # in the inference stage. Now, we assume it is streaming inference
            cos_tmp = cos.index_select(0, offset) # find the index oposition
            sin_tmp = sin.index_select(0, offset)
        else:
            cos_tmp = cos[:T]
            sin_tmp = sin[:T]
        q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos_tmp, sin_tmp) # add rope postional embedding
        k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos_tmp, sin_tmp) # rope_n_elem: 128
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1) # without any meaning?
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        # install the KV cache
        k, v, pos_k = self._complete_kv(k, v) #
        pos_k = pos_k.view(1, -1)
        pos_q = offset + torch.arange(T, device=q.device, dtype=torch.long).view(-1, 1)
        delta = pos_q - pos_k # for causal?
        attn_bias = (pos_k >= 0) & (delta >= 0)
        if self.config.context is not None:
            attn_bias = attn_bias & (delta < self.config.context)
        y = self.scaled_dot_product_attention(q, k, v, attn_bias) # attn_bias is the mask
        y = y.reshape(B, T, self.config.head_size * self.config.n_head)  # re-assemble all head outputs side by side
        if state is not None:
            state.offset.add_(T)
            state.offset_cpu += T
        # output projection
        return self.proj(y)

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "attn.weight": "attn.linear.weight",
            "attn.bias": "attn.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class GptNeoxMLP(lit_model.GptNeoxMLP):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.fc = LoRALinear(
            config.n_embd,
            config.intermediate_size,
            bias=config.bias,
            r=(config.lora_r if config.lora_mlp else 0),
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
        self.proj = LoRALinear(
            config.intermediate_size,
            config.n_embd,
            bias=config.bias,
            r=(config.lora_r if config.lora_mlp else 0),
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )

        self.config = config

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "fc.weight": "fc.linear.weight",
            "fc.bias": "fc.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class LLaMAMLP(lit_model.LLaMAMLP):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.fc_1 = LoRALinear(
            config.n_embd,
            config.intermediate_size,
            bias=config.bias,
            r=(config.lora_r if config.lora_mlp else 0),
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
        self.fc_2 = LoRALinear(
            config.n_embd,
            config.intermediate_size,
            bias=config.bias,
            r=(config.lora_r if config.lora_mlp else 0),
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
        self.proj = LoRALinear(
            config.intermediate_size,
            config.n_embd,
            bias=config.bias,
            r=(config.lora_r if config.lora_mlp else 0),
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )

        self.config = config

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "fc_1.weight": "fc_1.linear.weight",
            "fc_1.bias": "fc_1.linear.bias",
            "fc_2.weight": "fc_2.linear.weight",
            "fc_2.bias": "fc_2.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class GemmaMLP(LLaMAMLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
        return self.proj(x)


class LLaMAMoE(lit_model.LLaMAMoE):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.gate = LoRALinear(
            config.n_embd,
            config.n_expert,
            bias=False,
            r=(config.lora_r if config.lora_mlp else 0),
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
        self.experts = nn.ModuleList(LLaMAMLP(config) for _ in range(config.n_expert))

        self.config = config

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {"gate.weight": "gate.linear.weight"}
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def merge_lora_weights(model: GPT) -> None:
    """Merge LoRA weights into the full-rank weights to speed up inference."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()