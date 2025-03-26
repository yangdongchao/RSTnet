import json
import math
from typing import List, Tuple
import numpy as np
import torch
from torch import nn
from modules.seanet import SEANetEncoder, SEANetDecoder
import modules.transformer as Stransformer
import torch
import torch.nn as nn
from functools import partial
from vector_quantize_pytorch import ResidualVQ
import torch.nn.functional as F
from quantization.base import QuantizedResult

class SplitResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer with separate projections for the first quantizer and the rest.
    Args:
        n_q (int): Number of residual vector quantizers used.
        n_semantic_q (int): Number of residual vector quantizers used for the semantic quantizer.
        dimension: the dimension of codebook
        input_dimension: the dimension of input features
        no_quantization_mode (str): if 'true_skip', when doing no quantization, the input will not go
            through the sub quantizers. If `independent`, independent decisions are taken by
            the semantic and acoustic quantizers. If `same` (the default), the same decision is taken by both.
        **kwargs: Arguments to the constructor of `ResidualVectorQuantizer` that are shared between both.
    """

    def __init__(
        self,
        *,
        n_q: int = 8,
        dimension: int = 128,
        bins: int = 2048,
        input_dimension: int = 128,
        n_q_semantic: int = 1,
        **kwargs,
    ):
        super().__init__()
        assert n_q > n_q_semantic, (
            f"Number of quantizers {n_q} must be larger "
            f"than the number of semantic quantizers {n_q_semantic}."
        )
        self.max_n_q = n_q
        self.n_q_semantic = n_q_semantic
        self.n_q_acoustic = n_q - n_q_semantic
        self.rvq_first = ResidualVQ(
                dim = input_dimension,
                codebook_size = bins, # codebook size
                decay = 0.9, # the exponential moving average decay, lower means the dictionary will change faster
                commitment_weight = 1.,   # the weight on the commitment loss
                threshold_ema_dead_code = 2,
                use_cosine_sim = False,
                codebook_dim = dimension,
                num_quantizers= self.n_q_semantic,
            )
        self.rvq_rest = ResidualVQ(
                dim = input_dimension,
                codebook_size = bins, # codebook size
                decay = 0.9, # the exponential moving average decay, lower means the dictionary will change faster
                commitment_weight = 1.,   # the weight on the commitment loss
                threshold_ema_dead_code = 2,
                use_cosine_sim = False,
                codebook_dim = dimension,
                num_quantizers= self.n_q_acoustic,
            )

    def _renorm_and_add(
        self,
        first_val: torch.Tensor,
        rest_val: torch.Tensor,
        n_q_semantic: int,
        n_q_acoustic: int,
    ):
        """Renormalizes values from `rvq_first` and `rvq_rest` and adds them.

        This allows correcting statistics that are normalized by the number of quantizers. To renormalize, we use the
        number of quantizers that are actually used, e.g. taking into account quantizer dropout.
        """
        n_q = n_q_semantic + n_q_acoustic
        renorm_first_val = first_val * n_q_semantic / n_q
        renorm_rest_val = rest_val * n_q_acoustic / n_q
        return renorm_first_val + renorm_rest_val

    def cosine_similarity_loss(self, feature, target_feature):
        """
        feature: B, T, D
        target_feature: B, T ,D
        """
        n = min(feature.size(1), target_feature.size(1))
        distill_loss = - torch.log(torch.sigmoid(F.cosine_similarity(feature[:, :n], target_feature[:, :n], axis=1))).mean()
        return distill_loss

    def forward(self, x: torch.Tensor, frame_rate: int = None, semantic_features: torch.Tensor=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T] with `C` number of channels.
            frame_rate (int): frame rate of the input (e.g `T = frame_rate * duration`), used to compute
                the bandwidth.
            semantic_features: the semantic features from teacher model

        Returns:
            QuantizedResult: Quantized result with the following attributes:
                - `x` (torch.Tensor): Quantized tensor of shape [B, C, T].
                - `codes` (torch.Tensor): Quantized codes of shape [B, K, T] with `K` number of codebooks.
                - `bw` (torch.Tensor): Bandwidth of the quantized tensor in kbits per second.
                - `penalty` (torch.Tensor): Commitment loss.
                - `metrics` (dict): RVQ metrics, in particular rate of dead code replacement, and entropy.
        """
        # print('x ', x.shape)
        x = x.transpose(1,2) # transfer to B, T, C
        # print('self.rvq_first ', self.rvq_first)
        semantic_quantized_latent, semantic_indices, semantic_commitment_loss = self.rvq_first(x.clone()) # 
        # print('semantic_quantized_latent ', semantic_quantized_latent.shape, semantic_indices.shape, semantic_commitment_loss.shape)
        # print('semantic_features ', semantic_features.shape)
        # assert 1==2
        if semantic_features is not None:
            sim_loss = self.cosine_similarity_loss(semantic_quantized_latent, semantic_features)
        else:
            sim_loss = 0.0
        acoustic_quantized_latent, acoustic_indices, acoustic_commitment_loss = self.rvq_rest(x)
        full_quantized_emb = semantic_quantized_latent + acoustic_quantized_latent
        full_quantized_codes = torch.cat(
            [semantic_indices, acoustic_indices], dim=-1
        ) # B, T, N
        # This is the actual number of quantizers used,  e.g. taking into account quantizer dropout.
        n_q_semantic = semantic_indices.shape[-1]
        n_q_acoustic = acoustic_indices.shape[-1]
        #full_quantized_bandwidth = semantic_result.bandwidth + acoustic_result.bandwidth
        full_quantized_bandwidth = None
        full_quantized_penalty = self._renorm_and_add(
            semantic_commitment_loss.mean(), acoustic_commitment_loss.mean(), n_q_semantic, n_q_acoustic
        ) #  merge loss
        return QuantizedResult(
            full_quantized_emb,
            full_quantized_codes,
            full_quantized_bandwidth,
            penalty=full_quantized_penalty,
            metrics=None,
            sim_loss=sim_loss,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a given input tensor with the specified frame rate at the given bandwidth.
        The RVQ encode method sets the appropriate number of quantizer to use
        and returns indices for each quantizer.
        """
        x = x.transpose(1,2) # transfer to B, T, C
        semantic_quantized_latent, semantic_codec, semantic_commitment_loss = self.rvq_first(x)
        if self.max_n_q > self.n_q_semantic:
            acoustic_quantized_latent, acoustic_code, acoustic_commitment_loss = self.rvq_rest(x)
            codes = torch.cat([semantic_codec, acoustic_code], dim=-1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        # codes is [B, K, T], with T frames, K nb of codebooks.
        quantized=self.rvq_first.get_output_from_indices(codes[:, : ,:self.n_q_semantic])
        if codes.shape[1] > self.n_q_semantic:
            quantized+=self.rvq_rest.get_output_from_indices(codes[:,: ,self.n_q_semantic :])
        return quantized

    @property
    def total_codebooks(self):
        return self.max_n_q

    @property
    def num_codebooks(self):
        return self.rvq_first.num_codebooks + self.rvq_rest.num_codebooks

    @property
    def n_q(self):
        return self.rvq_first.n_q + self.rvq_rest.n_q

    @property
    def dimension(self):
        return self.rvq_first.dimension

    @property
    def semantic_quantizer(self):
        """This returns the quantizer that models the first level of the hierarchy (typically semantic)."""
        return self.rvq_first

    @property
    def acoustic_quantizer(self):
        """This returns the quantizer that models the higher levels of the hierarchy (typically acoustic)."""
        return self.rvq_rest

    def set_num_codebooks(self, n: int):
        assert n >= self.n_q_semantic and n <= self.total_codebooks
        self.rvq_rest.set_num_codebooks(n - self.n_q_semantic)

    @property
    def cardinality(self) -> int:
        assert self.rvq_rest.cardinality == self.rvq_first.cardinality
        return self.rvq_first.cardinality
