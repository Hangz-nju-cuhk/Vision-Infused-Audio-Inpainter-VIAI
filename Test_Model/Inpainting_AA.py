import numpy as np
import matplotlib

matplotlib.use('Agg')
from tqdm import tqdm
import torch
import os
from collections import OrderedDict
import librosa
from torch import optim
from nnmnkwii import preprocessing as P
from keras.utils import np_utils
from tensorboardX import SummaryWriter
from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw, is_scalar_input
from loss_functions import DiscretizedMixturelogisticLoss, MaskedCrossEntropyLoss, sequence_mask, \
    ExponentialMovingAverage, GANLoss
import loss_functions
from Test_Model import Whole_test
from wavenet_vocoder.mixture import sample_from_discretized_mix_logistic
from utils import audio, lrschedule, model_util, util
from wavenet_vocoder import WaveNet
from networks import Inpainting_Networks, Discriminator_Networks, Image_Embedding, New_Inpainting_Networks
import random
import torch.nn.functional as F
import Options_inpainting

hparams = Options_inpainting.Inpainting_Config()

if torch.cuda.is_available():
    device = torch.device("cuda")


class AudioModel(Whole_test.AudioModel):

    def __init__(self, hparams, device=device):
        Whole_test.AudioModel.__init__(self, hparams, device=device)
        self.Tensor = torch.Tensor

        self.device = device
        self.hparams = hparams
        self.name = hparams.name
        self.fs = hparams.sample_rate
        self.train = 0
        self.use_gan = 0
        self.update_wavenet = False
        self.decode_model = WaveNet(scalar_input=is_scalar_input(hparams.input_type))
        self.Mel_Encoder = Inpainting_Networks.MelEncoder(hparams=hparams)
        self.Mel_EncoderSync = Inpainting_Networks.MelEncoder(hparams=hparams)
        self.Mel_Decoder_Image = New_Inpainting_Networks.MelDecoderImage2(hparams=hparams, norm_layer=hparams.normlayer)
        self.VideoEncoder = Image_Embedding.ImageEmbedding2(hparams=hparams)
        self.netD = Discriminator_Networks.MelDiscriminator()
        self.Inpainting_Dis = Discriminator_Networks.Inpainting_Dis()
        self.L2Contrastive = loss_functions.L2ContrastiveLoss(margin=hparams.L2margin)
        # if hparams.load_pretrain:
        #     self.decode_model = model_util.load_pretrain_checkpoint(hparams.pretrain_path, self.decode_model)
        ###########********criterions*******#########################
        self.criterionL1 = torch.nn.L1Loss().to(device)
        self.criterionGAN = GANLoss(use_lsgan=False, device=device)
        if is_mulaw_quantize(hparams.input_type):
            self.audio_criterion = MaskedCrossEntropyLoss().to(device)
        else:
            self.audio_criterion = DiscretizedMixturelogisticLoss().to(device)

        self.criterionL2 = torch.nn.MSELoss().to(device)

        if hparams.exponential_moving_average:
            self.ema = ExponentialMovingAverage(hparams.ema_decay)
            for name, param in self.decode_model.named_parameters():
                if param.requires_grad:
                    self.ema.register(name, param.data)
        else:
            self.ema = None

        ##############********* use cuda ***********################
        self.model_parallel = False
        if hparams.model_parallel:
            self.model_parallel = True
            self.Mel_Encoder = torch.nn.DataParallel(self.Mel_Encoder)
            self.Mel_EncoderSync = torch.nn.DataParallel(self.Mel_EncoderSync)
            self.VideoEncoder = torch.nn.DataParallel(self.VideoEncoder)
            self.Mel_Decoder_Image = torch.nn.DataParallel(self.Mel_Decoder_Image)
            self.netD = torch.nn.DataParallel(self.netD)
            self.Inpainting_Dis = torch.nn.DataParallel(self.Inpainting_Dis)
            self.decode_model = torch.nn.DataParallel(self.decode_model)
        self.decode_model.to(device)
        self.Mel_Encoder.to(device)
        self.Mel_EncoderSync.to(device)
        self.VideoEncoder.to(device)
        self.Mel_Decoder_Image.to(device)
        self.Inpainting_Dis.to(device)
        self.netD.to(device)
        self.criterionL1.to(device)
        self.criterionGAN.to(device)
        self.L2Contrastive.to(device)

        ###########******** optimizer *******#########################
        if self.update_wavenet:
            param_groups = [{'params': self.decode_model.parameters(), 'lr_mult': 1.0},
                            {'params': self.Mel_Encoder.parameters(), 'lr_mult': 1.0},
                            {'params': self.Mel_EncoderSync.parameters(), 'lr_mult': 1.0},
                            {'params': self.VideoEncoder.parameters(), 'lr_mult': 1.0},
                            {'params': self.Mel_Decoder_Image.parameters(), 'lr_mult': 1.0}]
        else:
            param_groups = [{'params': self.Mel_Decoder_Image.parameters(), 'lr_mult': 1.0},
                            {'params': self.Mel_Encoder.parameters(), 'lr_mult': 1.0},
                            {'params': self.Mel_EncoderSync.parameters(), 'lr_mult': 1.0},
                            {'params': self.VideoEncoder.parameters(), 'lr_mult': 1.0}]
        self.optimizer_G = optim.Adam(param_groups,
                                      lr=hparams.initial_learning_rate, betas=(
                hparams.adam_beta1, hparams.adam_beta2),
                                      eps=hparams.adam_eps, weight_decay=hparams.weight_decay,
                                      amsgrad=hparams.amsgrad)

        ##############

        self.optimizer_D = torch.optim.Adam(list(self.netD.parameters()) + list(self.Inpainting_Dis.parameters()),
                                            lr=hparams.initial_learning_rate, betas=(hparams.adam_beta1, 0.999))

        self.blank_length = hparams.initial_blank_length

        self.clip_thresh = self.hparams.clip_thresh

        ##############
        if self.model_parallel:
            self.receptive_field = self.decode_model.module.receptive_field
        else:
            self.receptive_field = self.decode_model.receptive_field
        # self.receptive_field = self.decode_model.receptive_field

        print("Receptive field (samples / ms): {} / {}".format(
                self.receptive_field, self.receptive_field / self.fs * 1000))


    def test_inpainting(self):
        self.mel_y = self.mel_y.to(device)
        self.mel_x = self.mel_x.to(device)
        self.image_block = self.image_block.to(device)
        self.flow_block = self.flow_block.to(device)
        self.mask = torch.zeros_like(self.mel_x)
        self.mask[:,:,:, self.select_start: self.select_end] = 1
        self.mask = self.mask.to(device)
        self.inverse_mask = torch.ones_like(self.mel_x) - self.mask
        self.inverse_mask = self.inverse_mask.to(self.device)
        with torch.no_grad():
            mel_net = self.Mel_EncoderSync.forward(self.mel_y)

        inpainting_net = self.Mel_Encoder.forward(self.mel_x)

        x_size = self.mel_y.size()

        # self.video_net = self.VideoEncoder.forward(self.image_block, self.flow_block)
        # self.video_net = self.VideoEncoder.forward(self.image_block)


        self.audio_net = mel_net[-1].detach()

        # self.matching_L1 = self.criterionL1(self.video_net, self.audio_net) * self.hparams.lambda_B

        self.fake_mel = self.Mel_Decoder_Image(inpainting_net, x_size, self.audio_net)

    def forward_inpainting(self):
        self.mel_y = self.mel_y.to(device)
        self.mel_x = self.mel_x.to(device)
        self.image_block = self.image_block.to(device)
        self.flow_block = self.flow_block.to(device)
        self.mask = torch.zeros_like(self.mel_x)
        self.mask[:, :, :, self.select_start: self.select_end] = 1
        self.mask = self.mask.to(device)

        with torch.no_grad():
            mel_net = self.Mel_EncoderSync.forward(self.mel_y)

        inpainting_net = self.Mel_Encoder.forward(self.mel_x)

        x_size = self.mel_y.size()

        # self.video_net = self.VideoEncoder.forward(self.image_block, self.flow_block)
        # self.video_net = self.VideoEncoder.forward(self.image_block)


        self.audio_net = mel_net[-1].detach()

        # self.matching_L1 = self.criterionL1(self.video_net, self.audio_net) * self.hparams.lambda_B

        self.fake_mel = self.Mel_Decoder_Image(inpainting_net, x_size, self.audio_net)

        self.loss_mel_L1 = self.criterionL1(self.fake_mel * 255, self.mel_y * 255)

    def save_inpainting_checkpoint(self, global_step, global_test_step, checkpoint_dir, epoch, hparams=hparams):
        checkpoint_path = os.path.join(
                checkpoint_dir, self.name + "_checkpoint_step{:09d}.pth.tar".format(global_step))
        optimizer_G_state = self.optimizer_G.state_dict() if hparams.save_optimizer_state else None
        optimizer_D_state = self.optimizer_D.state_dict() if hparams.save_optimizer_state else None
        torch.save({
            "Mel_Encoder": self.Mel_Encoder.state_dict(),
            "decode_model": self.decode_model.state_dict(),
            "Mel_Decoder_Image": self.Mel_Decoder_Image.state_dict(),
            "Mel_EncoderSync": self.Mel_EncoderSync.state_dict(),
            "netD": self.netD.state_dict(),
            "Inpainting_Dis": self.Inpainting_Dis.state_dict(),
            # "discriminator_audio": self.discriminator_audio.state_dict(),
            "optimizer_G": optimizer_G_state,
            "optimizer_D": optimizer_D_state,
            "global_step": global_step,
            "global_epoch": epoch,
            "global_test_step": global_test_step,
        }, checkpoint_path)
        print("Saved checkpoint:", checkpoint_path)

    def load_inpainting_checkpoint(self, path, reset_optimizer):
        print("Load checkpoint from: {}".format(path))
        checkpoint = torch.load(path)
        self.Mel_Encoder = util.copy_state_dict(checkpoint["Mel_Encoder"], self.Mel_Encoder)
        self.Mel_EncoderSync = util.copy_state_dict(checkpoint["Mel_EncoderSync"], self.Mel_EncoderSync)
        self.Mel_Decoder_Image = util.copy_state_dict(checkpoint["Mel_Decoder_Image"], self.Mel_Decoder_Image)
        self.netD = util.copy_state_dict(checkpoint["netD"], self.netD)

        # self.Inpainting_Dis = util.copy_state_dict(checkpoint["Inpainting_Dis"], self.Inpainting_Dis)
        # self.decode_model = util.copy_state_dict(checkpoint["decode_model"], self.decode_model)
        # self.discriminator_audio = util.copy_state_dict(checkpoint["discriminator_audio"], self.discriminator_audio)
        # self.VideoEncoder = util.copy_state_dict(checkpoint["VideoEncoder"], self.VideoEncoder)
        if not reset_optimizer:
            optimizer_G_state = checkpoint["optimizer_G"]
            optimizer_D_state = checkpoint["optimizer_D"]
            if optimizer_G_state is not None:
                print("Load optimizer state from {}".format(path))
                self.optimizer_G.load_state_dict(optimizer_G_state)
            if optimizer_D_state is not None:
                self.optimizer_D.load_state_dict(optimizer_D_state)
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]
        global_test_step = checkpoint.get("global_test_step", 0)
        return global_step, global_epoch, global_test_step

    def load_part_checkpoint(self):
        # print("Load checkpoint from: {}".format(self.hparams.pretrain_path))
        # checkpoint = torch.load(self.hparams.pretrain_path)
        # self.Mel_EncoderSync = util.copy_state_dict(checkpoint["Mel_Encoder"], self.Mel_EncoderSync)
        # self.VideoEncoder = util.copy_state_dict(checkpoint["VideoEncoder"], self.VideoEncoder)

        print("Load checkpoint from: {}".format(self.hparams.wavenet_pretrain))
        checkpoint = torch.load(self.hparams.wavenet_pretrain)
        self.decode_model = util.copy_state_dict(checkpoint["model"], self.decode_model)