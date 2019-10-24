import torch
import torch.nn as nn
import Options_inpainting
import sys
import torch.nn.functional as F
sys.path.append('..')

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


class MelDecoder(nn.Module):
    def __init__(self, hparams=hparams, norm_layer=hparams.normlayer):
        super(MelDecoder, self).__init__()
        self.relu = nn.ReLU(True)
        self.hparams = hparams
        self.deconv1_1 = nn.ConvTranspose2d(256, 256, 3, 1, (0, 1))
        self.deconv1_1_bn = norm_layer(256)
        self.deconv1_2 = nn.ConvTranspose2d(256, 256, 3, 1, 1)
        self.deconv1_2_bn = norm_layer(256)
        self.convblock1 = TransConvBlock(256, 256, "1", nums=2, norm_layer=norm_layer)
        self.convblock2 = TransConvBlock(256, 128, "2", nums=3, norm_layer=norm_layer)
        self.convblock3 = TransConvBlock(128, 64, "3", nums=3, norm_layer=norm_layer)
        self.convblock4 = TransConvBlock(64 * 2, 32, "4", nums=3, norm_layer=norm_layer)
        self.convblock5 = TransConvBlock(32, 32, "5", nums=4, norm_layer=norm_layer)
        self.conv6_1 = nn.ConvTranspose2d(32, 32, 3, 1, 1)
        self.conv6_2 = nn.ConvTranspose2d(32, 1, 3, 1, 1)
        self.conv6_1_bn = norm_layer(32)
        self.relu = nn.ReLU(True)
        self.sig = nn.Sigmoid()
        self.orig_size = [hparams.cin_channels, hparams.max_mel_lengths]
        self.upsample_mode = 'bilinear'

    def forward(self, net, x_size):
        out = self.deconv1_1(net[-1])
        out = self.deconv1_1_bn(out)
        out = self.relu(out)
        out = self.deconv1_2(out)
        out = self.relu(self.deconv1_2_bn(out))
        for i in range(1, len(net)):

            out = F.interpolate(out, size=[net[-1 - i].size(2), net[-1 - i].size(3)], mode=self.upsample_mode, align_corners=True)
            # out = F.upsample(out, size=[net[-1 - i].size(2), net[-1 - i].size(3)], mode=self.upsample_mode, align_corners=True)
            if i == 3:
                out = torch.cat((out, net[-(i + 1)]), 1)
            out = self._modules['convblock' + str(i + 1)](out)
        out = F.interpolate(out, size=[x_size[2], x_size[3]], mode=self.upsample_mode, align_corners=True)
        # out = F.upsample(out, size=[x_size[2], x_size[3]], mode=self.upsample_mode, align_corners=True)
        out = self.conv6_1(out)
        out = self.relu(self.conv6_1_bn(out))
        out = self.conv6_2(out)
        out = self.sig(out)
        return out


class MelDecoderImage(nn.Module):
    def __init__(self, hparams=hparams, norm_layer=hparams.normlayer):
        super(MelDecoderImage, self).__init__()
        self.relu = nn.ReLU(True)
        self.hparams = hparams
        self.deconv1_1 = nn.ConvTranspose2d(256, 256, 3, 1, (0, 1))
        self.deconv1_1_bn = norm_layer(256)
        self.deconv1_1_1 = nn.ConvTranspose2d(256 * 2, 256, 3, 1, (0, 1))
        self.deconv1_1_1_bn = norm_layer(256)
        self.deconv1_2 = nn.ConvTranspose2d(256, 256, 3, 1, 1)
        self.deconv1_2_bn = norm_layer(256)
        self.convblock2 = TransConvBlock(256, 128, "2", nums=3, norm_layer=norm_layer)
        self.convblock3 = TransConvBlock(128, 64, "3", nums=3, norm_layer=norm_layer)
        self.convblock4 = TransConvBlock(64 * 2, 32, "4", nums=3, norm_layer=norm_layer)
        self.convblock5 = TransConvBlock(32, 32, "5", nums=4, norm_layer=norm_layer)
        self.conv6_1 = nn.ConvTranspose2d(32, 32, 3, 1, 1)
        self.conv6_2 = nn.ConvTranspose2d(32, 1, 3, 1, 1)
        self.conv6_1_bn = norm_layer(32)
        self.relu = nn.ReLU(True)
        self.sig = nn.Sigmoid()
        self.orig_size = [hparams.cin_channels, hparams.max_mel_lengths]
        self.upsample_mode = 'bilinear'


    def forward(self, net, x_size, video_net=None):
        # out = self.deconv1_1(net[-1])
        # out = self.deconv1_1_bn(out)
        video_net = video_net.view(net[-1].size(0), -1, net[-1].size(2), net[-1].size(3))
        input = torch.cat([net[-1], video_net], 1)
        out = self.deconv1_1_1(input)
        out = self.deconv1_1_1_bn(out)
        out = self.relu(out)
        out = self.deconv1_2(out)
        out = self.relu(self.deconv1_2_bn(out))
        for i in range(1, len(net)):
            out = F.interpolate(out, size=[net[-1 - i].size(2), net[-1 - i].size(3)], mode=self.upsample_mode, align_corners=True)
            # out = F.upsample(out, size=[net[-1 - i].size(2), net[-1 - i].size(3)], mode=self.upsample_mode, align_corners=True)
            if i == 3:
                out = torch.cat((out, net[-(i + 1)]), 1)

            out = self._modules['convblock' + str(i + 1)](out)
        out = F.interpolate(out, size=[x_size[2], x_size[3]], mode=self.upsample_mode, align_corners=True)
        out = self.conv6_1(out)
        out = self.relu(self.conv6_1_bn(out))
        out = self.conv6_2(out)
        out = self.sig(out)
        return out

    def init_deconv_1_1_1(self):
        deconv1weight = self.deconv1_1.weight.unsqueeze(0)
        deconv1weight = deconv1weight.expand(2, 256, 256, 3, 3).contiguous()
        self.deconv1_1_1.weight.data = deconv1weight.view(512, 256, 3, 3)


class MelDecoderImage2(nn.Module):
    def __init__(self, hparams=hparams, norm_layer=hparams.normlayer):
        super(MelDecoderImage2, self).__init__()
        self.relu = nn.ReLU(True)
        self.hparams = hparams
        self.deconv1_1 = nn.ConvTranspose2d(256, 256, 3, 1, (0, 1))
        self.deconv1_1_bn = norm_layer(256)
        self.deconv1_1_1 = nn.ConvTranspose2d(256 * 2, 256, 3, 1, (0, 1))
        self.deconv1_1_1_bn = norm_layer(256)
        self.deconv1_2 = nn.ConvTranspose2d(256, 256, 3, 1, 1)
        self.deconv1_2_bn = norm_layer(256)
        self.convblock2 = TransConvBlock(256, 128, "2", nums=3, norm_layer=norm_layer)
        self.convblock3 = TransConvBlock(128, 64, "3", nums=3, norm_layer=norm_layer)
        self.convblock4 = TransConvBlock(64, 32, "4", nums=3, norm_layer=norm_layer)
        self.convblock5 = TransConvBlock(32 * 2, 32, "5", nums=2, norm_layer=norm_layer)
        self.conv6_1 = nn.ConvTranspose2d(32, 32, 3, 1, 1)
        self.conv6_2 = nn.ConvTranspose2d(32, 1, 3, 1, 1)
        self.conv6_1_bn = norm_layer(32)
        self.relu = nn.ReLU(True)
        self.sig = nn.Sigmoid()
        self.orig_size = [hparams.cin_channels, hparams.max_mel_lengths]
        self.upsample_mode = 'bilinear'


    def forward(self, net, x_size, video_net=None):
        # out = self.deconv1_1(net[-1])
        # out = self.deconv1_1_bn(out)
        video_net = video_net.view(net[-1].size(0), -1, net[-1].size(2), net[-1].size(3))
        input = torch.cat([net[-1], video_net], 1)
        out = self.deconv1_1_1(input)
        out = self.deconv1_1_1_bn(out)
        out = self.relu(out)
        out = self.deconv1_2(out)
        out = self.relu(self.deconv1_2_bn(out))
        for i in range(1, len(net)):
            out = F.interpolate(out, size=[net[-1 - i].size(2), net[-1 - i].size(3)], mode=self.upsample_mode, align_corners=True)
            # out = F.upsample(out, size=[net[-1 - i].size(2), net[-1 - i].size(3)], mode=self.upsample_mode, align_corners=True)
            if i == 4:
                out = torch.cat((out, net[-(i + 1)]), 1)

            out = self._modules['convblock' + str(i + 1)](out)
        out = F.interpolate(out, size=[x_size[2], x_size[3]], mode=self.upsample_mode, align_corners=True)
        out = self.conv6_1(out)
        out = self.relu(self.conv6_1_bn(out))
        out = self.conv6_2(out)
        out = self.sig(out)
        return out

    def init_deconv_1_1_1(self):
        deconv1weight = self.deconv1_1.weight.unsqueeze(0)
        deconv1weight = deconv1weight.expand(2, 256, 256, 3, 3).contiguous()
        self.deconv1_1_1.weight.data = deconv1weight.view(512, 256, 3, 3)



class MelDecoder_old(nn.Module):
    def __init__(self, hparams=hparams, norm_layer=hparams.normlayer):
        super(MelDecoder_old, self).__init__()
        self.relu = nn.ReLU(True)
        self.hparams = hparams
        self.deconv1_1 = nn.ConvTranspose2d(256, 256, 3, 1, (0, 1))
        self.deconv1_1_bn = norm_layer(256)
        self.deconv1_2 = nn.ConvTranspose2d(256, 256, 3, 1, 1)
        self.deconv1_2_bn = norm_layer(256)
        self.convblock1 = TransConvBlock(256, 256, "1", nums=2, norm_layer=norm_layer)
        self.convblock2 = TransConvBlock(256, 128, "2", nums=3, norm_layer=norm_layer)
        self.convblock3 = TransConvBlock(128, 64, "3", nums=3, norm_layer=norm_layer)
        self.convblock4 = TransConvBlock(64 , 32, "4", nums=3, norm_layer=norm_layer)
        self.convblock5 = TransConvBlock(32 * 2, 32, "5", nums=2, norm_layer=norm_layer)
        self.conv6_1 = nn.ConvTranspose2d(32, 32, 3, 1, 1)
        self.conv6_2 = nn.ConvTranspose2d(32, 1, 3, 1, 1)
        self.conv6_1_bn = norm_layer(32)
        self.relu = nn.ReLU(True)
        self.sig = nn.Sigmoid()
        self.orig_size = [hparams.cin_channels, hparams.max_mel_lengths]
        self.upsample_mode = 'bilinear'

    def forward(self, net, x_size):
        out = self.deconv1_1(net[-1])
        out = self.deconv1_1_bn(out)
        out = self.relu(out)
        out = self.deconv1_2(out)
        out = self.relu(self.deconv1_2_bn(out))
        for i in range(1, len(net)):

            out = F.interpolate(out, size=[net[-1 - i].size(2), net[-1 - i].size(3)], mode=self.upsample_mode, align_corners=True)
            # out = F.upsample(out, size=[net[-1 - i].size(2), net[-1 - i].size(3)], mode=self.upsample_mode, align_corners=True)
            if i == 4:
                out = torch.cat((out, net[-(i + 1)]), 1)
            out = self._modules['convblock' + str(i + 1)](out)
        out = F.interpolate(out, size=[x_size[2], x_size[3]], mode=self.upsample_mode, align_corners=True)
        # out = F.upsample(out, size=[x_size[2], x_size[3]], mode=self.upsample_mode, align_corners=True)
        out = self.conv6_1(out)
        out = self.relu(self.conv6_1_bn(out))
        out = self.conv6_2(out)
        out = self.sig(out)
        return out


