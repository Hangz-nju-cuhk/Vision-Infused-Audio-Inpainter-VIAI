# coding: utf-8
import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from .modules import Embedding

from .modules import Conv1d1x1, EncodeConv
from .mixture import sample_from_discretized_mix_logistic
import Config

hparams = Config.Config()


def _expand_global_features(B, T, g, bct=True):
    """Expand global conditioning features to all time steps

    Args:
        B (int): Batch size.
        T (int): Time length.
        g (Tensor): Global features, (B x C) or (B x C x 1).
        bct (bool) : returns (B x C x T) if True, otherwise (B x T x C)

    Returns:
        Tensor: B x C x T or B x T x C or None
    """
    if g is None:
        return None
    g = g.unsqueeze(-1) if g.dim() == 2 else g
    if bct:
        g_bct = g.expand(B, -1, T)
        return g_bct.contiguous()
    else:
        g_btc = g.expand(B, -1, T).transpose(1, 2)
        return g_btc.contiguous()


def receptive_field_size(total_layers, num_cycles, kernel_size,
                         dilation=lambda x: 2**x):
    """Compute receptive field size

    Args:
        total_layers (int): total layers
        num_cycles (int): cycles
        kernel_size (int): kernel size
        dilation (lambda): lambda to compute dilation factor. ``lambda x : 1``
          to disable dilated convolution.

    Returns:
        int: receptive field size in sample

    """
    assert total_layers % num_cycles == 0
    layers_per_cycle = total_layers // num_cycles
    dilations = [dilation(i % layers_per_cycle) for i in range(total_layers)]
    return (kernel_size - 1) * sum(dilations) + 1


class WaveNetEncoder(nn.Module):
    """The WaveNet model that supports local and global conditioning.

    Args:
        out_channels (int): Output channels. If input_type is mu-law quantized
          one-hot vecror. this must equal to the quantize channels. Other wise
          num_mixtures x 3 (pi, mu, log_scale).
        layers (int): Number of total layers
        stacks (int): Number of dilation cycles
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        skip_out_channels (int): Skip connection channels.
        kernel_size (int): Kernel size of convolution layers.
        dropout (float): Dropout probability.

        weight_normalization (bool): If True, DeepVoice3-style weight
          normalization is applied.
        scalar_input (Bool): If True, scalar input ([-1, 1]) is expected, otherwise
          quantized one-hot vector is expected.
    """

    def __init__(self, out_channels=128, layers=24, stacks=4,
                 residual_channels=128,
                 gate_channels=128,
                 skip_out_channels=128, ae_bottleneck_width=128,
                 kernel_size=3, dropout=1 - 0.95,
                 weight_normalization=True,
                 scalar_input=True,
                 ):
        super(WaveNetEncoder, self).__init__()
        self.scalar_input = scalar_input
        self.layers = layers
        assert layers % stacks == 0
        self.layers_per_stack = layers // stacks
        if scalar_input:
            self.first_conv = Conv1d1x1(1, residual_channels)
        else:
            self.first_conv = Conv1d1x1(out_channels, residual_channels)

        for layer in range(layers):
            dilation = 2**(layer % self.layers_per_stack)
            self.add_module('dilation' + str(layer), EncodeConv(residual_channels, gate_channels,
                                                                kernel_size=kernel_size, skip_out_channels=skip_out_channels,
                                                                bias=True, dilation=dilation, dropout=dropout,
                                                                weight_normalization=weight_normalization))
        self.last_conv_layer = \
            Conv1d1x1(residual_channels, ae_bottleneck_width,
                      weight_normalization=weight_normalization),

        self.avgpool = torch.nn.AvgPool1d(640)

        self.receptive_field = receptive_field_size(layers, stacks, kernel_size)

    def forward(self, x, softmax=False):
        """Forward step

        Args:
            x (Tensor): One-hot encoded audio signal, shape (B x C x T)
            softmax (bool): Whether applies softmax or not.

        Returns:
            Tensor: output, shape B x out_channels x T
        """
        B, _, T = x.size()

        # Feed data to network
        net = self.first_conv(x)

        for layer in range(self.layers//4):
            net = self._modules['dilation' + str(layer)](net)
        net1 = net
        for layer in range(self.layers//4, self.layers//4 * 2):
            net1 = self._modules['dilation' + str(layer)](net1)
        net2 = net1
        for layer in range(self.layers//3 * 2, self.layers):
            net2 = self._modules['dilation' + str(layer)](net2)

        net3 = self.last_conv_layer(net2)

        x = F.softmax(net3, dim=1) if softmax else net3

        return x
