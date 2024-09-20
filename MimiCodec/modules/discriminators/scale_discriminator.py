from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d
from torch.nn.utils import weight_norm, spectral_norm

from modules.commons.base_layers import Conv1d


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, config):
        super(MultiScaleDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList()
        for i in range(config.num_scales):
            use_spec_norm= True if i==0 else False
            self.discriminators.append(
                ScaleDiscriminator(
                    use_spectral_norm=use_spec_norm,
                    use_cond=config.get('use_cond', False),
                    upsample_rates=config.get('hop_size', 240) // (2**i),
                    num_mels=config.get('num_mels', 80),
                )
            )

        self.pools = nn.ModuleList()
        for i in range(config.num_scales - 1):
            self.pools.append(
                AvgPool1d(
                    kernel_size=config.pool_kernel_size,
                    stride=config.pool_stride,
                    padding=int(config.pool_stride / 2 + 0.5)))

    def forward(self, y, y_hat, mel=None, **kwargs):
        real_outputs = []
        fake_outputs = []
        real_feature_maps = []
        fake_feature_maps = []
        for i, d in enumerate(self.discriminators):
            real_out, real_feat_map = d(y, mel)
            fake_out, fake_feat_map = d(y_hat, mel)
            real_outputs.append(real_out)
            fake_outputs.append(fake_out)
            real_feature_maps.append(real_feat_map)
            fake_feature_maps.append(fake_feat_map)
            # down-sampling
            y = self.pools[i-1](y)
            y_hat = self.pools[i-1](y_hat)

        return real_outputs, fake_outputs, real_feature_maps, fake_feature_maps


class ScaleDiscriminator(torch.nn.Module):
    def __init__(
        self,
        use_spectral_norm: bool = False,
        use_cond: bool = False,
        upsample_rates: int = None,
        num_mels: int = 80,
    ):
        super(ScaleDiscriminator, self).__init__()
        norm_func = weight_norm if use_spectral_norm == False else spectral_norm
        self.use_cond = use_cond

        input_dim = 1
        if use_cond:
            t = upsample_rates
            self.cond_net = torch.nn.ConvTranspose1d(num_mels, 1, t * 2, stride=t, padding=t // 2)
            input_dim = 2

        self.convs = nn.ModuleList([
            norm_func(Conv1d(input_dim, 128, 15, 1, padding=7)),
            norm_func(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_func(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_func(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_func(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_func(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_func(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.final_conv = norm_func(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(
        self,
        x: torch.Tensor,
        mel: Optional[torch.Tensor] = None,
    ):
        if self.use_cond:
            x_mel = self.cond_net(mel)
            x = torch.cat([x_mel, x], 1)

        feature_map = []
        for i, l in enumerate(self.convs):
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            feature_map.append(x)
        x = self.final_conv(x)
        feature_map.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feature_map
