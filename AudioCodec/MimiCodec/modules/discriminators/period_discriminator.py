from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d
from torch.nn.utils import weight_norm, spectral_norm

from modules.commons.ops import get_padding


# Ref: https://github.com/jik876/hifi-gan
class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, config):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList()
        for i in range(len(config.period_sizes)):
            self.discriminators.append(
                PeriodDiscriminator(
                    config.period_sizes[i],
                    kernel_size=config.period_kernel_size,
                    use_cond=config.get('use_cond', False),
                    hop_size=config.get('hop_size', 240),
                    num_mels=config.get('num_mels', 80),
                ),
            )

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

        return real_outputs, fake_outputs, real_feature_maps, fake_feature_maps


class PeriodDiscriminator(torch.nn.Module):
    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
        use_cond: bool = False,
        hop_size: int = 240,
        num_mels: int = 80,
    ):
        super(PeriodDiscriminator, self).__init__()
        self.use_cond = use_cond
        self.period = period
        norm_func = weight_norm if use_spectral_norm == False else spectral_norm

        input_dim = 1
        if use_cond:
            input_dim = 2
            self.cond_net = torch.nn.ConvTranspose1d(
                num_mels, 1, hop_size * 2, stride=hop_size, padding=hop_size // 2)

        self.convs = nn.ModuleList([
            norm_func(Conv2d(input_dim, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_func(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_func(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_func(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_func(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.final_conv = norm_func(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(
        self,
        x: torch.Tensor,
        mel: Optional[torch.Tensor] = None,
    ):
        feature_map = []

        if self.use_cond:
            x_mel = self.cond_net(mel)
            x = torch.cat([x_mel, x], 1)

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for i, l in enumerate(self.convs):
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            feature_map.append(x)

        x = self.final_conv(x)
        feature_map.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feature_map
