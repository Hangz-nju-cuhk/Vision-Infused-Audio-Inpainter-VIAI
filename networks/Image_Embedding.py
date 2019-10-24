import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo
import cv2
from Data_loaders import AV_loader
from Options_inpainting import Inpainting_Config
from utils import util
from networks.ResNet import BasicBlock, Bottleneck

hparams = Inpainting_Config()

class ResNet(nn.Module):

    def __init__(self, block, layers, channel_size=3, length_feature=hparams.length_feature):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(channel_size, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, length_feature)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ImageResnet18(hparams=hparams):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if hparams.resnet_pretrain:
        pretrain = torch.load(hparams.resnet_pretrain_path)
        util.copy_state_dict(pretrain, model)
    return model


def FlowResnet18(hparams=hparams):
    model = ResNet(BasicBlock, [2, 2, 2, 2], channel_size=2, length_feature=hparams.length_feature)
    if hparams.resnet_pretrain:
        pretrain = torch.load(hparams.resnet_pretrain_path)
        model = util.copy_state_dict(pretrain, model)
        conv1_weight = pretrain["conv1.weight"].data
        flow_conv1_weight = conv1_weight.mean(1)
        flow_conv1_weight = flow_conv1_weight.unsqueeze(1)
        flow_conv1_weight = flow_conv1_weight.expand([64, 2, 7, 7]).contiguous()
        model.conv1.weight.data = flow_conv1_weight
    return model


class ImageEmbedding(nn.Module):

    def __init__(self, hparams=hparams):
        super(ImageEmbedding, self).__init__()
        self.hparams = hparams
        self.image_single_model = ImageResnet18(hparams)
        self.flow_single_model = FlowResnet18(hparams)
        self.conv_1 = torch.nn.Conv1d(2*hparams.length_feature, 2*hparams.length_feature, 3, 2, 1, bias=False)
        self.bn_1 = nn.BatchNorm1d(2*hparams.length_feature)
        self.conv_2 = torch.nn.Conv1d(2*hparams.length_feature, hparams.length_feature, 3, 2, 1, bias=False)
        self.bn_2 = nn.BatchNorm1d(hparams.length_feature)
        self.relu = nn.ReLU(True)

    def forward(self, video_block, flow_block):
        input_image = video_block.view(-1, 3, self.hparams.image_size, self.hparams.image_size)
        input_flow = flow_block.view(-1, 2, self.hparams.image_size, self.hparams.image_size)
        image_out = self.image_single_model(input_image)
        flow_out = self.flow_single_model(input_flow)
        image_out = image_out.view(video_block.size(0), -1, self.hparams.length_feature)
        flow_out = flow_out.view(video_block.size(0), -1, self.hparams.length_feature)
        fea_cat = torch.cat((image_out, flow_out), 2)
        fea_cat = torch.transpose(fea_cat, 2, 1)
        out = self.conv_1(fea_cat)
        self.relu(self.bn_1(out))
        out = self.conv_2(out)
        out = out.unsqueeze(2)
        return out


class ImageEmbedding_single(nn.Module):

    def __init__(self, hparams=hparams, image=1):
        super(ImageEmbedding_single, self).__init__()
        self.image = image
        self.hparams = hparams
        self.image_single_model = ImageResnet18(hparams) if image else FlowResnet18(hparams)
        self.conv_1 = torch.nn.Conv1d(hparams.length_feature, hparams.length_feature, 3, 2, 1, bias=False)
        self.bn_1 = nn.BatchNorm1d(hparams.length_feature)
        self.conv_2 = torch.nn.Conv1d(hparams.length_feature, hparams.length_feature, 3, 2, 1, bias=False)
        self.bn_2 = nn.BatchNorm1d(hparams.length_feature)
        self.relu = nn.ReLU(True)

    def forward(self, video_block):
        input_image = video_block.view(-1, 3 if self.image else 2, self.hparams.image_size, self.hparams.image_size)
        image_out = self.image_single_model(input_image)
        image_out = image_out.view(video_block.size(0), -1, self.hparams.length_feature)
        image_out = image_out.transpose(1, 2)
        out = self.conv_1(image_out)
        self.relu(self.bn_1(out))
        out = self.conv_2(out)
        return out

class ImageEmbedding_finetune(nn.Module):

    def __init__(self, hparams=hparams, image=1):
        super(ImageEmbedding_finetune, self).__init__()
        self.image = image
        self.hparams = hparams
        self.conv_1 = torch.nn.Conv1d(hparams.length_feature, hparams.length_feature, 3, 2, 1, bias=False)
        self.bn_1 = nn.BatchNorm1d(hparams.length_feature)
        self.conv_2 = torch.nn.Conv1d(hparams.length_feature, hparams.length_feature, 3, 2, 1, bias=False)
        self.bn_2 = nn.BatchNorm1d(hparams.length_feature)
        self.relu = nn.ReLU(True)

    def forward(self, image_out):

        image_out = image_out.transpose(1, 2)
        out = self.conv_1(image_out)
        self.relu(self.bn_1(out))
        out = self.conv_2(out)
        out = out.unsqueeze(2)
        return out


class ImageEmbedding2(nn.Module):

    def __init__(self, hparams=hparams):
        super(ImageEmbedding2, self).__init__()
        self.hparams = hparams
        self.image_single_model = ImageResnet18(hparams)
        self.flow_single_model = FlowResnet18(hparams)
        self.conv_1 = torch.nn.Conv1d(2*hparams.length_feature, 2*hparams.length_feature, 3, 2, 1, bias=False)
        self.bn_1 = nn.BatchNorm1d(2*hparams.length_feature)
        self.conv_2 = torch.nn.Conv1d(2*hparams.length_feature, hparams.length_feature, 3, 2, 1, bias=False)
        self.bn_2 = nn.BatchNorm1d(hparams.length_feature)
        self.relu = nn.ReLU(True)

    def forward(self, video_block, flow_block):
        input_image = video_block.view(-1, 3, self.hparams.image_size, self.hparams.image_size)
        input_flow = flow_block.view(-1, 2, self.hparams.image_size, self.hparams.image_size)
        image_out = self.image_single_model(input_image)
        flow_out = self.flow_single_model(input_flow)
        image_out = image_out.view(video_block.size(0), -1, self.hparams.length_feature)
        flow_out = flow_out.view(video_block.size(0), -1, self.hparams.length_feature)
        fea_cat = torch.cat((image_out, flow_out), 2)
        fea_cat = torch.transpose(fea_cat, 2, 1)
        out = self.conv_1(fea_cat)
        # self.relu(self.bn_1(out))
        out = self.conv_2(out)
        out = out.unsqueeze(2)
        return out, fea_cat
