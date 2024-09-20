import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from omegaconf.dictconfig import DictConfig

from modules.commons.base_layers import Conv1d
from modules.commons.pqmf import PQMF


""" Collaborative multi-band discriminator (CoMBD) and Sub-Band discriminator (SBD)
introduced in the paper `Avocodo: Generative Adversarial Network for Artifact-free Vocoder`.
"""


class CoMBD(nn.Module):
    """ Collaborative multi-band discriminator proposed in the paper
    `Avocodo: Generative Adversarial Network for Artifact-free Vocoder`.

    """
    def __init__(
        self, filters, kernels, groups, strides, use_spectral_norm=False
    ):
        super().__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList()
        init_channel = 1
        for i, (f, k, g, s) in enumerate(zip(filters, kernels, groups, strides)):
            self.convs.append(
                norm_f(Conv1d(init_channel, f, k, stride=s, groups=g)))
            init_channel = f
        self.conv_post = norm_f(Conv1d(filters[-1], 1, kernel_size=3))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MDC(nn.Module):
    """ Multiscale Dilated Convolution Block proposed in the paper
    `Neural Photo Editing with Introspective Adversarial Networks`.
    """
    def __init__(
        self, in_channel, channel, kernel, stride, dilations, use_spectral_norm=False
    ):
        super().__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = torch.nn.ModuleList()
        self.num_dilations = len(dilations)
        for d in dilations:
            self.convs.append(
                norm_f(
                    Conv1d(in_channel, channel, kernel, stride=1, dilation=d)
                )
            )
        self.conv_out = norm_f(Conv1d(channel, channel, 3, stride=stride))

    def forward(self, x):
        xs = None
        for l in self.convs:
            if xs is None:
                xs = l(x)
            else:
                xs += l(x)

        x = xs / self.num_dilations

        x = self.conv_out(x)
        x = F.leaky_relu(x, 0.1)
        return x


class SubBandDiscriminator(torch.nn.Module):
    """ Sub-module in the sub-band discriminator in Avocodo.
    """
    def __init__(self, init_channel, channels, kernel, strides, dilations, use_spectral_norm=False):
        super(SubBandDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        # Multiscale Dilated Convolution modules
        self.mdcs = torch.nn.ModuleList()
        for c, s, d in zip(channels, strides, dilations):
            self.mdcs.append(MDC(init_channel, c, kernel, s, d))
            init_channel = c
        self.conv_post = norm_f(Conv1d(init_channel, 1, 3, ))

    def forward(self, x):
        fmap = []

        for l in self.mdcs:
            x = l(x)
            fmap.append(x)
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


# ========================================== #
# ========          CoMBD       ============ #
# ========================================== #
class MultiCoMBDiscriminator(nn.Module):
    """
    Config:
        kernels
        channels
        groups
        strides
    """
    def __init__(self, config: DictConfig):
        super().__init__()
        
        self.combd_1 = CoMBD(
            filters=config.channels,
            kernels=config.kernels[0],
            groups=config.groups,
            strides=config.strides
        )
        self.combd_2 = CoMBD(
            filters=config.channels,
            kernels=config.kernels[1],
            groups=config.groups,
            strides=config.strides
        )
        self.combd_3 = CoMBD(
            filters=config.channels,
            kernels=config.kernels[2],
            groups=config.groups,
            strides=config.strides
        )
        self.pqmf_2 = PQMF(subbands=2, taps=256, beta=10.0)
        self.pqmf_4 = PQMF(subbands=4, taps=192, beta=10.0)

    def forward(self, x, x_hat, subband_outputs, **kwargs):
        x2_hat, x1_hat = subband_outputs
        y = []
        y_hat = []
        fmap = []
        fmap_hat = []

        p3, p3_fmap = self.combd_3(x)
        y.append(p3)
        fmap.append(p3_fmap)

        p3_hat, p3_fmap_hat = self.combd_3(x_hat)
        y_hat.append(p3_hat)
        fmap_hat.append(p3_fmap_hat)

        x2_ = self.pqmf_2.analysis(x)[:, :1, :]  # Select first band
        x1_ = self.pqmf_4.analysis(x)[:, :1, :]  # Select first band

        x2_hat_ = self.pqmf_2.analysis(x_hat)[:, :1, :]
        x1_hat_ = self.pqmf_4.analysis(x_hat)[:, :1, :]

        p2_, p2_fmap_ = self.combd_2(x2_)
        y.append(p2_)
        fmap.append(p2_fmap_)

        p2_hat_, p2_fmap_hat_ = self.combd_2(x2_hat)
        y_hat.append(p2_hat_)
        fmap_hat.append(p2_fmap_hat_)

        p1_, p1_fmap_ = self.combd_1(x1_)
        y.append(p1_)
        fmap.append(p1_fmap_)

        p1_hat_, p1_fmap_hat_ = self.combd_1(x1_hat)
        y_hat.append(p1_hat_)
        fmap_hat.append(p1_fmap_hat_)

        p2, p2_fmap = self.combd_2(x2_)
        y.append(p2)
        fmap.append(p2_fmap)

        p2_hat, p2_fmap_hat = self.combd_2(x2_hat_)
        y_hat.append(p2_hat)
        fmap_hat.append(p2_fmap_hat)

        p1, p1_fmap = self.combd_1(x1_)
        y.append(p1)
        fmap.append(p1_fmap)

        p1_hat, p1_fmap_hat = self.combd_1(x1_hat_)
        y_hat.append(p1_hat)
        fmap_hat.append(p1_fmap_hat)

        return y, y_hat, fmap, fmap_hat


# ========================================== #
# ========           SBD        ============ #
# ========================================== #
class MultiSubBandDiscriminator(nn.Module):
    """
    Config:
        tkernels
        fkernel
        tchannels
        fchannels
        tstrides
        fstride
        tdilations
        fdilations
        tsubband
        num_tsbd_bands
        num_fsbd_bands
        freq_init_ch
    """

    def __init__(self, config: DictConfig):

        super().__init__()

        self.fsbd = SubBandDiscriminator(
            init_channel=config.freq_init_ch,
            channels=config.fchannels,
            kernel=config.fkernel,
            strides=config.fstride,
            dilations=config.fdilations
        )
        self.tsubband1 = config.tsubband[0]
        self.tsbd1 = SubBandDiscriminator(
            init_channel=self.tsubband1,
            channels=config.tchannels,
            kernel=config.tkernels[0],
            strides=config.tstrides[0],
            dilations=config.tdilations[0]
        )
        self.tsubband2 = config.tsubband[1]
        self.tsbd2 = SubBandDiscriminator(
            init_channel=self.tsubband2,
            channels=config.tchannels,
            kernel=config.tkernels[1],
            strides=config.tstrides[1],
            dilations=config.tdilations[1]
        )
        self.tsubband3 = config.tsubband[2]
        self.tsbd3 = SubBandDiscriminator(
            init_channel=self.tsubband3,
            channels=config.tchannels,
            kernel=config.tkernels[2],
            strides=config.tstrides[2],
            dilations=config.tdilations[2]
        )
        self.pqmf_tsbd = PQMF(subbands=config.num_tsbd_bands, taps=256, cutoff_ratio=0.03, beta=10.0) # n
        self.pqmf_fsbd = PQMF(subbands=config.num_fsbd_bands, taps=256, cutoff_ratio=0.1, beta=9.0) # m

    def forward(self, x, x_hat, **kwargs):
        fmap = []
        fmap_hat = []
        y = []
        y_hat = []

        # Time analysis
        xn = self.pqmf_tsbd.analysis(x)
        xn_hat = self.pqmf_tsbd.analysis(x_hat)

        q3, feat_q3 = self.tsbd3(xn[:, :self.tsubband3, :])
        q3_hat, feat_q3_hat = self.tsbd3(xn_hat[:, :self.tsubband3, :])
        y.append(q3)
        y_hat.append(q3_hat)
        fmap.append(feat_q3)
        fmap_hat.append(feat_q3_hat)

        q2, feat_q2 = self.tsbd2(xn[:, :self.tsubband2, :])
        q2_hat, feat_q2_hat = self.tsbd2(xn_hat[:, :self.tsubband2, :])
        y.append(q2)
        y_hat.append(q2_hat)
        fmap.append(feat_q2)
        fmap_hat.append(feat_q2_hat)

        q1, feat_q1 = self.tsbd1(xn[:, :self.tsubband1, :])
        q1_hat, feat_q1_hat = self.tsbd1(xn_hat[:, :self.tsubband1, :])
        y.append(q1)
        y_hat.append(q1_hat)
        fmap.append(feat_q1)
        fmap_hat.append(feat_q1_hat)

        # Frequency analysis
        xm = self.pqmf_fsbd.analysis(x)
        xm_hat = self.pqmf_fsbd.analysis(x_hat)
        # print(xm.shape)
        # print(xm_hat.shape)

        xm = xm.transpose(-2, -1)
        xm_hat = xm_hat.transpose(-2, -1)

        q4, feat_q4 = self.fsbd(xm)
        q4_hat, feat_q4_hat = self.fsbd(xm_hat)
        y.append(q4)
        y_hat.append(q4_hat)
        fmap.append(feat_q4)
        fmap_hat.append(feat_q4_hat)

        return y, y_hat, fmap, fmap_hat
