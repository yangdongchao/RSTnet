import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from omegaconf import DictConfig


class MultiResolutionDiscriminator(torch.nn.Module):
    def __init__(self, config: DictConfig):
        super(MultiResolutionDiscriminator, self).__init__()
        self.resolutions = eval(config.resolutions)
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(config, resolution) for resolution in self.resolutions]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor, **kwargs):
        real_outputs = []
        fake_outputs = []
        real_feature_maps = []
        fake_feature_maps = []

        for d in self.discriminators:
            real_out, real_feat_map = d(y)
            fake_out, fake_feat_map = d(y_hat)
            real_outputs.append(real_out)
            fake_outputs.append(fake_out)
            real_feature_maps.append(real_feat_map)
            fake_feature_maps.append(fake_feat_map)

        return real_outputs, fake_outputs, real_feature_maps, fake_feature_maps


class DiscriminatorR(torch.nn.Module):
    def __init__(self, config, resolution):
        super(DiscriminatorR, self).__init__()

        self.resolution = resolution
        self.LRELU_SLOPE = config.lReLU_slope

        norm_f = weight_norm if config.use_spectral_norm == False else spectral_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        feature_map = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            feature_map.append(x)
        x = self.conv_post(x)
        feature_map.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feature_map

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
        x = x.squeeze(1)
        x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False) #[B, F, TT, 2]
        mag = torch.norm(x, p=2, dim =-1) #[B, F, TT]

        return mag
