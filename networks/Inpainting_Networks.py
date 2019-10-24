import torch
import torch.nn as nn
import Options_inpainting
import sys
import torch.nn.functional as F
from networks import Discriminator_Networks
sys.path.append('..')
from Data_loaders import mel_loader

hparams = Options_inpainting.Inpainting_Config()


class TransConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, name, nums=3,
                 kernel_size=3, padding=(1, 1), stride=(1, 1), norm_layer=nn.BatchNorm2d):
        super(TransConvBlock, self).__init__()
        self.nums = nums
        use_bias = norm_layer == nn.InstanceNorm2d
        self.relu = nn.ReLU(True)
        if isinstance(name, str):
            self.name = name
        else:
            raise Exception("name should be str")
        for i in range(self.nums):
            self.add_module('conv' + self.name + "_" + str(i), nn.ConvTranspose2d(inplanes, outplanes, padding=padding, kernel_size=kernel_size, stride=stride, bias=use_bias))
            self.add_module('conv' + self.name + "_" + str(i) + "_bn", norm_layer(outplanes))
            inplanes = outplanes
        self.initial()

    def forward(self, x):
        net = x
        for i in range(self.nums):
            net = self._modules['conv' + self.name + "_" + str(i)](net)
            net = self._modules['conv' + self.name + "_" + str(i) + "_bn"](net)
            net = self.relu(net)
        return net

    def initial(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class MelEncoder(nn.Module):
    def __init__(self, hparams=hparams, norm_layer=nn.BatchNorm2d):
        super(MelEncoder, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        self.relu = nn.LeakyReLU(0.2, True)
        self.hparams = hparams
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=use_bias)
        self.bn1 = norm_layer(32, affine=True)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), stride=(2, 1), padding=(1, 1), bias=use_bias)
        self.bn2 = norm_layer(64, affine=True)
        self.conv3 = nn.Conv2d(64, 128, (3, 3), (2, 2), (1, 1), bias=use_bias)
        self.bn3 = norm_layer(128, affine=True)
        self.conv4 = nn.Conv2d(128, 256, (3, 3), (2, 2), (1, 1), bias=use_bias)
        self.bn4 = norm_layer(256, affine=True)
        self.conv5 = nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), bias=use_bias)
        self.bn5 = norm_layer(256, affine=True)
        self.avgpool = nn.AvgPool2d((3, 1))
        self.initial()


    def forward(self, c):
        c = c.view(c.size(0), 1, self.hparams.cin_channels, -1)
        net = []
        middle = self.conv1(c)
        net.append(self.relu(self.bn1(middle)))  # receptive 3
        for i in range(1, 5):
            middle = self._modules['conv' + str(i + 1)](net[i - 1])
            net.append(self.relu(self._modules['bn' + str(i + 1)](middle)))
        net[-1] = self.avgpool(net[-1])
        return net

    def initial(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

