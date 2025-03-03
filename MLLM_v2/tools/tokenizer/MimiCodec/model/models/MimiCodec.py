import json
import math
from typing import List, Tuple
import numpy as np
import torch
from torch import nn
from tools.tokenizer.MimiCodec.model.modules.seanet import SEANetEncoder, SEANetDecoder
import tools.tokenizer.MimiCodec.model.modules.transformer as Stransformer
from tools.tokenizer.MimiCodec.model.quantization.vq import SplitResidualVectorQuantizer
from tools.tokenizer.MimiCodec.model.modules.resample import ConvDownsample1d, ConvTrUpsample1d
import torch
import torch.nn as nn
from functools import partial

class Semantic_linear_pool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ln_layer = nn.Linear(in_channels, out_channels)
        self.pl = nn.AvgPool1d(kernel_size=8, stride=4)
    def forward(self, x):
        x = self.ln_layer(x)
        x = self.pl(x.transpose(1,2))
        return x.transpose(1, 2)

class MimiCodec(nn.Module):
    def __init__(
        self,
        sample_rate=24000,
        n_filters=64,
        encoder_rates=[4, 5, 6, 8],
        compress = 2,
        causal = True,
        latent_dim=512,
        codebook_size=4096,
        codebook_dim=32,
        rvq_layers = 8,
        num_heads = 8,
        num_layers = 8,
        layer_scale = 0.01,
        context = 250,
        dim_feedforward = 2048,
        semantic_feature_dim = 1024,
        target_frame_rate = 12.5,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        _seanet_kwargs = { "channels": 1, "dimension": latent_dim, "causal": causal, "n_filters": n_filters, "n_residual_layers": 1, 
                "activation": "ELU", "compress": compress, "dilation_base": 2, "disable_norm_outer_blocks": 0, "kernel_size": 7, 
                "residual_kernel_size": 3, "last_kernel_size": 3, "norm": "none", "pad_mode": "constant", "ratios": encoder_rates, "true_skip": True
        } 
        _quantizer_kwargs = { "dimension": codebook_dim, "n_q": rvq_layers, "bins": codebook_size,
            "input_dimension": _seanet_kwargs["dimension"], "output_dimension": _seanet_kwargs["dimension"]
        }
        _transformer_kwargs = {
            "d_model": _seanet_kwargs["dimension"], "num_heads": num_heads, "num_layers": num_layers, "causal": causal, "layer_scale": layer_scale,
            "context": context, "conv_layout": True, "max_period": 10000, "gating": "none", "norm": "layer_norm", "positional_embedding": "rope",
            "dim_feedforward": 2048, "input_dimension": _seanet_kwargs["dimension"], "output_dimensions": [_seanet_kwargs["dimension"]],
        }
        self.encoder = SEANetEncoder(**_seanet_kwargs)
        self.decoder = SEANetDecoder(**_seanet_kwargs)
        self.hop_length = encoder_rates[0]*encoder_rates[1]*encoder_rates[2]*encoder_rates[3]
        self.encoder_frame_rate = 24000/self.hop_length
        self.target_frame_rate = target_frame_rate # our target
        self.learnt = True
        self.downsample = ConvDownsample1d(int(self.encoder_frame_rate/self.target_frame_rate), dimension=latent_dim, learnt=self.learnt, causal=causal)
        self.upsample = ConvTrUpsample1d(int(self.encoder_frame_rate/self.target_frame_rate), dimension=latent_dim, learnt=self.learnt, causal=causal, channel_wise=True)
        self.semantic_mapping_layer = Semantic_linear_pool(semantic_feature_dim, latent_dim)
        
        self.encoder_transformer = Stransformer.ProjectedTransformer(**_transformer_kwargs)
        self.decoder_transformer = Stransformer.ProjectedTransformer(**_transformer_kwargs)
        self.quantizer = SplitResidualVectorQuantizer(**_quantizer_kwargs)

    def forward(self, audio_data: torch.Tensor, semantic_features: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        length = audio_data.shape[-1]
        z = self.encoder(audio_data) # encode the audio
        z = self.encoder_transformer(z)[0] # 
        z = self.downsample(z) 
        frame_rate = int(z.shape[-1] / (length/self.sample_rate))
        semantic_features = self.semantic_mapping_layer(semantic_features)
        quantizedResult = self.quantizer(z, frame_rate, semantic_features) # x: torch.Tensor codes: torch.Tensor bandwidth: torch.Tensor penalty: tp.Optional[torch.Tensor] = Non metrics: dict = field(default_factory=dict)
        """
        random choose the feature into decoder, either quantized features or non-quantized features.
        apply quantization 60% of quantization features
        """
        mask = torch.rand((audio_data.shape[0])) >= 0.4 # transfer to true or false
        mask = mask.unsqueeze(1).unsqueeze(2).repeat(1, quantizedResult.x.shape[1], quantizedResult.x.shape[2]).to(audio_data.device)
        z_q = torch.where(mask, quantizedResult.x, z) # 
        z_q = self.upsample(z_q)
        z_q = self.decoder_transformer(z_q)[0]
        rec = self.decoder(z_q)
        return rec[..., :length], quantizedResult.codes, quantizedResult.penalty, quantizedResult.sim_loss

    def encode(self, audio_data: torch.Tensor) -> List[torch.Tensor]:
        length = audio_data.shape[-1]
        import time
        st_time = time.time()
        z = self.encoder(audio_data) # encode the audio
        z = self.encoder_transformer(z)[0] # 
        z = self.downsample(z) 
        quantizedResult = self.quantizer.encode(z)
        return quantizedResult

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        import time
        st_time = time.time()
        z_q = self.quantizer.decode(codes)
        z_q = self.upsample(z_q)
        z_q = self.decoder_transformer(z_q)[0]
        rec = self.decoder(z_q)
        return rec

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        model = cls(**config)
        return model
