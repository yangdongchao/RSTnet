# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from einops import rearrange
import torch
from torch import nn
from torch import distributed
import torch.nn.functional as F
import torch.distributed as dist

class SyncFunction(torch.autograd.Function):
    @staticmethod
    # @torch.no_grad()
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]

def _is_complex_or_float(tensor):
    return torch.is_floating_point(tensor) or torch.is_complex(tensor)

def all_reduce(tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM):
    if _is_distributed():
        return torch.distributed.all_reduce(tensor, op)

def world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1

def _check_number_of_params(params: tp.List[torch.Tensor]):
    # utility function to check that the number of params in all workers is the same,
    # and thus avoid a deadlock with distributed all reduce.
    if not _is_distributed() or not params:
        return
    #print('params[0].device ', params[0].device)
    tensor = torch.tensor([len(params)], device=params[0].device, dtype=torch.long)
    all_reduce(tensor)
    if tensor.item() != len(params) * world_size():
        # If not all the workers have the same number, for at least one of them,
        # this inequality will be verified.
        raise RuntimeError(f"Mismatch in number of params: ours is {len(params)}, "
                           "at least one worker has a different one.")

def broadcast_tensors(tensors: tp.Iterable[torch.Tensor], src: int = 0):
    """Broadcast the tensors from the given parameters to all workers.
    This can be used to ensure that all workers have the same model to start with.
    """
    if not _is_distributed():
        return
    tensors = [tensor for tensor in tensors if _is_complex_or_float(tensor)]
    _check_number_of_params(tensors)
    handles = []
    for tensor in tensors:
        handle = torch.distributed.broadcast(tensor.data, src=src, async_op=True)
        handles.append(handle)
    for handle in handles:
        handle.wait()

class _CodebookForwardResult(tp.NamedTuple):
    quantized: torch.Tensor
    codes: torch.Tensor
    metrics: tp.Dict[str, torch.Tensor]


class _VQForwardResult(tp.NamedTuple):
    quantized: torch.Tensor
    codes: torch.Tensor
    loss: torch.Tensor
    metrics: tp.Dict[str, torch.Tensor]


def _ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float) -> None:
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def _uniform_init(*shape: int) -> torch.Tensor:
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def _sample_vectors(samples: torch.Tensor, num: int) -> torch.Tensor:
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def _compute_entropy(usage: torch.Tensor) -> torch.Tensor:
    # Usage is some unnormalized distribution.
    proba = usage / usage.sum()
    p_log_p = torch.where(
        proba == 0, zero_scalar(usage.device), proba * torch.log(proba)
    )
    return -p_log_p.sum()


def _is_distributed() -> bool:
    # Checks if we need to use distributed routines.
    return distributed.is_initialized() and distributed.get_world_size() > 1


def zero_scalar(device) -> torch.Tensor:
    """Returns a 0. value on the given device without introducing a synchronization point."""
    return torch.zeros([1], device=device)[0]

def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10, frames_to_use: int = 10_000):
    """ Run K-means clustering on samples.
    Args:
        samples (tensor): shape [B * T, D]
        num_clusters (int): number of centroids.
        num_iters (int): number of iterations.
    """
    dim, dtype = samples.shape[-1], samples.dtype

    if frames_to_use < samples.shape[0]:
        samples = sample_vectors(samples, frames_to_use)

    # Init means
    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(
            means, "c d -> () c d"
        )
        dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 50,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        replaced_usage_ratio: float = 0.1,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = _uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        
        # Flag variable to indicate whether the codebook is initialized
        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        # Runing EMA cluster size/count: N_i^t in eq. (6) in vqvae paper
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        # Codebook
        self.register_buffer("embed", embed)
        # EMA codebook: eq. (7) in vqvae paper
        self.register_buffer("embed_avg", embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        """ Initialize codebook.
        Args:
            data (tensor): [B * T, D].
        """
        if self.inited:
            return
        
        if dist.is_available() and dist.is_initialized():
            # [B * T * world_size, D]
            data = SyncFunction.apply(data)

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        # if is_primary():
            # print(batch_samples.shape)

        ## NOTE (snippet added by Songxiang Liu): gather data from all gpus
        if _is_distributed():
            # [B * T * world_size, D]
            batch_samples = SyncFunction.apply(batch_samples)

        # if is_primary():
            # print(batch_samples.shape)

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        broadcast_tensors(self.buffers())

    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x) # [B, T, D] -> [B*T, D]
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x, initialize=True):
        # shape: [B, T, D]
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x) # [B, T, D] -> [B*T, D]
        
        # Initialize codebook
        self.init_embed_(x)

        embed_ind = self.quantize(x) # [B*T,]
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype) # [B*T, cb-size]
        embed_ind = self.postprocess_emb(embed_ind, shape) # [B, T]
        quantize = self.dequantize(embed_ind) # [B, T, D]

        if self.training:
            ### Update codebook by EMA
            embed_onehot_sum = embed_onehot.sum(0) # [cb-size,]
            embed_sum = x.t() @ embed_onehot # [D, cb-size]
            if _is_distributed():
                dist.all_reduce(embed_onehot_sum)
                dist.all_reduce(embed_sum)
            # Update ema cluster count N_i^t, eq. (6) in vqvae paper
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            # Update ema embed: eq. (7) in vqvae paper
            self.embed_avg.data.mul_(self.decay).add_(embed_sum.t(), alpha=1 - self.decay)
            # apply laplace smoothing
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.epsilon) / (n + self.codebook_size * self.epsilon) * n
            )
            # Update ema embed: eq. (8) in vqvae paper
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            self.expire_codes_(x)
        metrics: tp.Dict[str, torch.Tensor] = {}
        return _CodebookForwardResult(quantize, embed_ind, metrics)


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.

    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_usage_ratio (float): Defines the threshold for the cluster usage under which a centroid
            is replaced. This is expressed as a fraction of the usage a centroid would get under
            a uniform distribution, so that it doesn't depend on the batch size etc.
        replaced_usage_ratio (float): When replacing a centroid, use this as an initial centroid usage,
            to avoid the centroid getting replaced too quickly.
        check_unused_every (int): Check for unused centroids every `check_unused_every` iterations.
            This is to avoid too many synchronization points.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_usage_ratio: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        if codebook_dim is None:
            codebook_dim = dim

        requires_projection = codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        )
        self.epsilon = epsilon
        self._codebook = EuclideanCodebook(
            dim=codebook_dim,
            codebook_size=codebook_size,
            decay=decay,
            epsilon=epsilon,
            **kwargs,
        )
        self.codebook_size = codebook_size

    @property
    def embedding(self):
        return self._codebook.embed # 

    def _rearrange_input(self, x):
        x = rearrange(x, "b d n -> b n d")
        return x

    def _rearrange_output(self, quantized):
        quantized = rearrange(quantized, "b n d -> b d n")
        return quantized

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes `x` into discrete integer codes."""
        x = self._rearrange_input(x)
        x = self.project_in(x)
        codes = self._codebook.encode(x)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Converts integer codes into quantized vectors."""
        quantized = self._codebook.decode(codes)
        quantized = self.project_out(quantized)
        quantized = self._rearrange_output(quantized)
        return quantized

    def forward(self, x: torch.Tensor, initialize: bool = True) -> _VQForwardResult:
        x = self._rearrange_input(x)
        quantized, codes, metrics = self._codebook(x, initialize=initialize)

        loss = zero_scalar(x.device)
        if self.training: # add for training. copy the gradient
            quantized = x + (quantized - x).detach()
        quantized = self.project_out(quantized)
        quantized = self._rearrange_output(quantized)

        return _VQForwardResult(quantized, codes, loss, metrics)


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.

    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, *, num_quantizers: int, codebook_offset: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )
        self.codebook_offset = codebook_offset

    def forward(
        self, x: torch.Tensor, n_q: tp.Optional[int] = None
    ) -> _VQForwardResult:
        """
        Args:
            x (torch.Tensor): input tensor to quantize, of shape `[B, C, T]`.
            n_q (int or None): if provided, number of codebook levels to use in RVQ.
        """

        quantized_out = zero_scalar(x.device)
        residual = x

        all_losses = []
        all_codes = []
        all_metrics: tp.Dict[str, torch.Tensor] = {}

        n_q = n_q or len(self.layers)
        previous_layer_is_initialized = True

        for i, layer in enumerate(self.layers[:n_q]):  # type: ignore
            quantized, codes, loss, metrics = layer(
                residual, initialize=previous_layer_is_initialized
            )

            if self.training:
                residual = residual - quantized 
                quantized_out = quantized_out + quantized
            else:
                quantized = quantized.detach() # remove for training
                residual = residual - quantized
                quantized_out = quantized_out + quantized
            
            all_codes.append(codes)
            all_losses.append(loss)

            for key, value in metrics.items():
                if key in all_metrics:
                    all_metrics[key] += value / n_q
                else:
                    all_metrics[key] = value / n_q
                all_metrics[key + f"_{i + self.codebook_offset}"] = value

        out_losses, out_codes = map(torch.stack, (all_losses, all_codes))
        return _VQForwardResult(quantized_out, out_codes, out_losses, all_metrics)

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        """Encodes `x` into discrete integer codes. If `n_q` is provided, only uses the first `n_q` codebook levels."""
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:  # type: ignore
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Converts the integer codes into quantized vectors."""
        quantized = zero_scalar(codes.device)
        for idx, layer_codes in enumerate(codes):
            layer = self.layers[idx]
            quantized = quantized + layer.decode(layer_codes)
        return quantized
