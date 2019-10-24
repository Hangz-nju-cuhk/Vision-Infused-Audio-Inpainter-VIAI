from __future__ import print_function, division
import torch
import torch.nn as nn
import Options_inpainting

hparams = Options_inpainting.Inpainting_Config()


class MelDiscriminator(nn.Module):
    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=True):
        super(MelDiscriminator, self).__init__()
        self.n_layers = n_layers
        self.use_sigmoid = True
        use_bias = norm_layer == nn.InstanceNorm2d
        self.relu = nn.LeakyReLU(0.2, True)

        self.conv1 = nn.Conv2d(input_nc, ndf, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), bias=use_bias)
        self.bn1 = norm_layer(ndf)
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            self.add_module('conv2_' + str(n), nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=(3, 3), stride=2, padding=1, bias=use_bias))
            self.add_module('norm_' + str(n), norm_layer(ndf * nf_mult))
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)

        self.conv3 = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.norm3 = norm_layer(ndf * nf_mult)
        self.conv4 = nn.Conv2d(ndf * nf_mult, 1,
                      kernel_size=3, stride=1, padding=1, bias=use_bias)
        if use_sigmoid:
            self.sig = nn.Sigmoid()

    def forward(self, input):
        net = self.conv1(input)
        netn = self.relu(self.bn1(net))
        for n in range(1, self.n_layers):
            netn = self._modules['conv2_' + str(n)](netn)
            netn = self._modules['norm_' + str(n)](netn)
            netn = self.relu(netn)
        net = self.conv3(netn)
        net = self.norm3(net)
        net = self.relu(net)
        net = self.conv4(net)
        if self.use_sigmoid:
            net = self.sig(net)
        return net


class Inpainting_Dis(nn.Module):
    def __init__(self):
        super(Inpainting_Dis, self).__init__()
        self.mel_conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.mel_bn1 = nn.BatchNorm2d(64)
        self.mel_conv2 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.mel_bn2 = nn.BatchNorm2d(128)
        self.mel_conv3 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)

        self.mel_bn3 = nn.BatchNorm2d(256)
        self.mel_conv4 = nn.Conv2d(256, 256, (10, 1), 1, bias=False)
        self.vid_conv1 = nn.Conv1d(512, 256, 3, 2, 1, bias=False)

        self.vid_bn1 = nn.BatchNorm1d(256)
        self.conv = nn.Conv1d(512, 1, 6, bias=False)
        self.relu = nn.LeakyReLU(0.2, True)
        self.sig = nn.Sigmoid()

    def forward(self, mel_inpainting, fea_inpainting):
        mel_net = self.mel_conv1(mel_inpainting)
        mel_net = self.relu(self.mel_bn1(mel_net))
        mel_net = self.mel_conv2(mel_net)
        mel_net = self.relu(self.mel_bn2(mel_net))
        mel_net = self.mel_conv3(mel_net)
        mel_net = self.relu(self.mel_bn3(mel_net))

        mel_net = self.mel_conv4(mel_net)
        vid_net = self.vid_conv1(fea_inpainting)
        vid_net = self.relu(self.vid_bn1(vid_net))
        mel_net = mel_net.squeeze(2)
        net = torch.cat((mel_net, vid_net), dim=1)
        net = self.conv(net)
        net = net.squeeze(1)
        net = self.sig(net)
        return net


class DomainDis(nn.Module):
    def __init__(self, hparams=hparams):
        super(DomainDis, self).__init__()
        self.conv1 = nn.Conv1d(hparams.length_feature, 256, 13, 1, 0, bias=False)
        self.relu = nn.ReLU(True)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        input = input.view(-1, hparams.length_feature, 13)
        out = self.conv1(input)
        out = self.relu(out)
        out = out.view(-1, 256)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.sig(out)
        return out



