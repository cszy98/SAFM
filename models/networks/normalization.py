"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm


class DepthConv(nn.Module):
    def __init__(self, fmiddle, kw=3, padding=1, stride=1):
        super().__init__()

        self.kw = kw
        self.stride = stride
        self.unfold = nn.Unfold(kernel_size=(self.kw,self.kw), dilation=1, padding=1, stride=stride)
        if True:
            BNFunc = nn.SyncBatchNorm
        else:
            BNFunc = nn.BatchNorm2d

        self.norm_layer = BNFunc(fmiddle, affine=True)
        
    def forward(self, x, conv_weights):

        N, C, H, W = x.size()
        
        conv_weights = conv_weights.view(N * C, self.kw * self.kw, H//self.stride, W//self.stride)
        #conv_weights = nn.functional.softmax(conv_weights, dim=1)
        x = self.unfold(x).view(N * C, self.kw * self.kw, H//self.stride, W//self.stride)
        x = torch.mul(conv_weights, x).sum(dim=1, keepdim=False).view(N, C, H//self.stride, W//self.stride)

        #x = self.norm_layer(x)

        return x


# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        pw = ks // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc+36, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')

        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SAFM(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        self.label_nc = label_nc

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        pw = ks // 2

        self.pre_seg = nn.Sequential(
            nn.Conv2d(label_nc-72, 36, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.pre_dis = nn.Sequential(
            nn.Conv2d(72, 36, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.gen_weights1 = nn.Sequential(
                    nn.Conv2d(36,36, kernel_size=3, padding=1), 
                    nn.ReLU(), 
                    nn.Conv2d(36, 36*9, kernel_size=3, padding=1))

        self.gen_weights2 = nn.Sequential(
                    nn.Conv2d(36,36, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(36, 36*9, kernel_size=3, padding=1))


        self.depconv1=DepthConv(36)
        self.depconv2=DepthConv(36)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc+36, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        
        pure_seg = segmap[:,:self.label_nc-72,:,:]
        pure_dis = segmap[:,self.label_nc-72:,:,:]
        
        pre_seg_rst = self.pre_seg(pure_seg)
        pre_dis_rst = self.pre_dis(pure_dis)
        seg_weights1 = self.gen_weights1(pre_seg_rst)
        seg_weights2 = self.gen_weights2(pre_seg_rst)
        dcov_dis1 = self.depconv1(pre_dis_rst,seg_weights1)
        dcov_dis2 = self.depconv2(dcov_dis1,seg_weights2)
        dcov_dis_final = torch.cat((pure_dis, dcov_dis2),dim=1)

        segmap = torch.cat((pure_seg, dcov_dis_final),dim=1)

        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta

        return out
