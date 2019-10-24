import numpy as np
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
import torch
import librosa
import os
from collections import OrderedDict
import torch.nn.functional as F
from nnmnkwii import preprocessing as P
from keras.utils import np_utils
import random
from Test_Model import Whole_test
from torch import optim
import loss_functions
from networks import Inpainting_Networks, Discriminator_Networks, Image_Embedding, New_Inpainting_Networks
from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw, is_scalar_input
from loss_functions import DiscretizedMixturelogisticLoss, MaskedCrossEntropyLoss, sequence_mask, ExponentialMovingAverage, GANLoss
from wavenet_vocoder.mixture import sample_from_discretized_mix_logistic
from wavenet_vocoder import WaveNet
from utils import audio, lrschedule, util, model_util
import Options_inpainting

hparams = Options_inpainting.Inpainting_Config()

if torch.cuda.is_available():
    device = torch.device("cuda")


class AudioModel(Whole_test.AudioModel):

    def __init__(self, hparams, device=device):
        Whole_test.AudioModel.__init__(self, hparams, device=device)
        self.Tensor = torch.Tensor
        self.train = 1
        self.use_gan = 0
        self.device = device
        self.hparams = hparams
        self.fs = hparams.sample_rate
        self.update_wavenet = False
        self.decode_model = WaveNet(scalar_input=is_scalar_input(hparams.input_type))
        self.Mel_Encoder = Inpainting_Networks.MelEncoder(hparams=hparams)
        self.Mel_Decoder_Image = New_Inpainting_Networks.MelDecoderImage(hparams=hparams, norm_layer=hparams.normlayer)
        self.netD = Discriminator_Networks.MelDiscriminator()
        self.Inpainting_Dis = Discriminator_Networks.Inpainting_Dis()
        self.VideoEncoder = Image_Embedding.ImageEmbedding2(hparams=hparams)
        self.discriminator_audio = Discriminator_Networks.DomainDis()
        # self.ResNet = Image_Embedding.ImageResnet18(hparams=hparams)
        # self.ImageEmbedding_finetune = Image_Embedding.ImageEmbedding_finetune()
        ###########********criterions*******#########################

        self.criterionL1 = torch.nn.L1Loss().to(device)
        self.criterionGAN = GANLoss(use_lsgan=False, device=device)
        self.L2Contrastive = loss_functions.L2ContrastiveLoss(margin=hparams.L2margin)

        if hparams.exponential_moving_average:
            self.ema = ExponentialMovingAverage(hparams.ema_decay)
            for name, param in self.decode_model.named_parameters():
                if param.requires_grad:
                    self.ema.register(name, param.data)
        else:
            self.ema = None

        if is_mulaw_quantize(hparams.input_type):
            self.audio_criterion = MaskedCrossEntropyLoss().to(device)
        else:
            self.audio_criterion = DiscretizedMixturelogisticLoss().to(device)


        ##############********* use cuda ***********################
        self.model_parallel = False
        if hparams.model_parallel:
            self.model_parallel = True
            self.Mel_Encoder = torch.nn.DataParallel(self.Mel_Encoder)
            self.netD = torch.nn.DataParallel(self.netD)
            self.VideoEncoder = torch.nn.DataParallel(self.VideoEncoder)
            self.Mel_Decoder_Image = torch.nn.DataParallel(self.Mel_Decoder_Image)
            self.Inpainting_Dis = torch.nn.DataParallel(self.Inpainting_Dis)
            # self.ResNet = torch.nn.DataParallel(self.ResNet)
            self.decode_model = torch.nn.DataParallel(self.decode_model)
            # self.ImageEmbedding_finetune = torch.nn.DataParallel(self.ImageEmbedding_finetune)
        self.decode_model.to(device)
        self.Mel_Encoder.to(device)
        self.netD.to(device)
        self.VideoEncoder.to(device)
        self.Mel_Decoder_Image.to(device)
        # self.ResNet.to(device)
        # self.ImageEmbedding_finetune.to(device)
        self.Inpainting_Dis.to(device)
        self.criterionL1.to(device)
        self.criterionGAN.to(device)
        self.L2Contrastive.to(device)

        # if hparams.checkpoint_restore_parts is not None:
        #     self.load_check_parts()

        ###########******** optimizer *******#########################
        if self.update_wavenet:
            param_groups = [{'params': self.decode_model.parameters(), 'lr_mult': 1.0},
                            {'params': self.Mel_Encoder.parameters(), 'lr_mult': 1.0},
                            {'params': self.Mel_Decoder_Image.parameters(), 'lr_mult': 1.0},
                            {'params': self.VideoEncoder.parameters(), 'lr_mult': 1.0}]
        else:
            param_groups = [{'params': self.Mel_Decoder_Image.parameters(), 'lr_mult': 1.0},
                            {'params': self.Mel_Encoder.parameters(), 'lr_mult': 1.0},
                            {'params': self.VideoEncoder.parameters(), 'lr_mult': 1.0}]

        self.optimizer_G = optim.Adam(param_groups,
                               lr=hparams.initial_learning_rate, betas=(
                                hparams.adam_beta1, hparams.adam_beta2),
                               eps=hparams.adam_eps, weight_decay=hparams.weight_decay,
                               amsgrad=hparams.amsgrad)

        ##############

        self.optimizer_D = torch.optim.Adam(list(self.netD.parameters()) + list(self.Inpainting_Dis.parameters()),
                                        lr=hparams.initial_learning_rate , betas=(hparams.adam_beta1, 0.999))

        self.clip_thresh = self.hparams.clip_thresh

        self.blank_length = hparams.initial_blank_length

        if self.model_parallel:
            self.receptive_field = self.decode_model.module.receptive_field
        else:
            self.receptive_field = self.decode_model.receptive_field
        # self.receptive_field = self.decode_model.receptive_field

        print("Receptive field (samples / ms): {} / {}".format(
                self.receptive_field, self.receptive_field / self.fs * 1000))


    def test_inpainting(self):

        self.mel_y = self.mel_y.to(self.device)
        self.mel_x = self.mel_x.to(self.device)
        self.image_block = self.image_block.to(self.device)
        self.flow_block = self.flow_block.to(self.device)
        self.mask = torch.zeros_like(self.mel_x)
        self.mask[:,:,:, self.select_start: self.select_end] = 1
        self.inverse_mask = torch.ones_like(self.mel_x) - self.mask
        self.mask = self.mask.to(device)
        self.inverse_mask = self.inverse_mask.to(self.device)
        mel_net = self.Mel_Encoder.forward(self.mel_y)

        inpainting_net = self.Mel_Encoder.forward(self.mel_x)

        x_size = self.mel_y.size()

        self.video_net, self.vid_fea = self.VideoEncoder.forward(self.image_block, self.flow_block)
        # self.video_net = self.VideoEncoder.forward(self.image_block)


        self.audio_net = mel_net[-1].detach()

        self.matching_L1 = self.criterionL1(self.video_net, self.audio_net) * self.hparams.lambda_B

        self.fake_mel = self.Mel_Decoder_Image(inpainting_net, x_size, self.video_net)

        self.fake_mel = self.mask * self.fake_mel + self.inverse_mask * self.mel_y

        no_need_list = [self.video_net, self.image_block, self.flow_block, self.VideoEncoder, self.Mel_Decoder_Image]

        for values in no_need_list:
            del values

    def forward_inpainting(self):

        self.mel_y = self.mel_y.to(self.device)
        self.mel_x = self.mel_x.to(self.device)
        self.image_block = self.image_block.to(self.device)
        self.flow_block = self.flow_block.to(self.device)
        self.mask = torch.zeros_like(self.mel_x)
        self.mask[:,:,:, self.select_start: self.select_end] = 1
        self.inverse_mask = torch.ones_like(self.mel_x) - self.mask
        self.mask = self.mask.to(device)
        self.inverse_mask = self.inverse_mask.to(self.device)
        mel_net = self.Mel_Encoder.forward(self.mel_y)

        inpainting_net = self.Mel_Encoder.forward(self.mel_x)

        x_size = self.mel_y.size()

        self.video_net, self.vid_fea = self.VideoEncoder.forward(self.image_block, self.flow_block)
        # self.video_net = self.VideoEncoder.forward(self.image_block)


        self.audio_net = mel_net[-1].detach()

        self.matching_L1 = self.criterionL1(self.video_net, self.audio_net) * self.hparams.lambda_B

        self.fake_mel = self.Mel_Decoder_Image(inpainting_net, x_size, self.video_net)

        self.fake_mel = self.mask * self.fake_mel + self.inverse_mask * self.mel_y


        self.mel_net_norm = util.l2_norm(self.audio_net.view(-1, self.hparams.length_feature * 13))
        self.video_net_norm = util.l2_norm(self.video_net.view(-1, self.hparams.length_feature * 13))

        self.mel_net_norm_buffer = self.mel_net_norm.detach()
        self.video_net_norm_buffer = self.video_net_norm.detach()

        self.EmbeddingL2_video = self.L2Contrastive.forward(self.video_net_norm, self.mel_net_norm_buffer)
        self.EmbeddingL2_audio = self.L2Contrastive.forward(self.mel_net_norm, self.video_net_norm_buffer)
        self.loss_mel_L1 = self.criterionL1(self.fake_mel * 255, self.mel_y * 255) * 5

        self.fake_audio_c = self.fake_mel.view(self.audio_c.size(0), self.audio_c.size(1), self.audio_c.size(2))[:self.hparams.wavenet_batch, :,
                        self.gen_start:self.gen_end]