import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from modules import *

class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        self.padding = self.receptive_field // 2
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups
        '''
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        '''
        self.conv = Conv1D(
            in_channels, out_channels, kernel_size,
            padding=self.padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x,params):
        # params here
        out = self.conv(x,params)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = Conv1D(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def forward(self, x,params):
        'normally residual is just x (if in channel -= out channel. Otherwise projection'

        if self.projector is None:
            residual = x
        else:
            cov_proj_params = get_child_dict(params, 'projector')
            residual = self.projector(x,cov_proj_params)

        x = F.gelu(x)
        # params here
        conv1_params = get_child_dict(params,'conv1')
        conv2_params = get_child_dict(params, 'conv2')

        x = self.conv1(x,conv1_params)
        x = F.gelu(x)
        x = self.conv2(x,conv2_params)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = Sequential(OrderedDict([
            ('Conv_{}'.format(str(i)),ConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                final=(i == len(channels) - 1)
            ))
            for i in range(len(channels))
        ]))

    def forward(self, x,params):
        # params here
        if params is None:
            out = self.net(x)
        else:
            out = self.net(x,params)
        return out

    def get_output_from_each_layer(self, x):
        'gets outputs from intermediate layers (with differnet receptive fields'
        child_nns = self.net.children()

        list_outputs = []
        input = x
        for net in child_nns:
            out = net(input)
            list_outputs.append(out.transpose(1, 2).cpu().numpy())
            input = out
        return list_outputs