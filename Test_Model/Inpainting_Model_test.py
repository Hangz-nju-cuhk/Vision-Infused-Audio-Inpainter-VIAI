import numpy as np
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
import torch
import os
import librosa
from collections import OrderedDict
import torch.nn.functional as F
import random
from keras.utils import np_utils
from nnmnkwii import preprocessing as P
from Test_Model import Whole_test
from torch import optim
from loss_functions import GANLoss
from networks import Inpainting_Networks, Discriminator_Networks, New_Inpainting_Networks, Image_Embedding
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

        self.device = device
        self.hparams = hparams
        self.fs = hparams.sample_rate
        self.train = 0
        self.update_wavenet = False
        self.decode_model = WaveNet(scalar_input=is_scalar_input(hparams.input_type))
        self.Mel_Encoder = Inpainting_Networks.MelEncoder(hparams=hparams, norm_layer=hparams.normlayer)
        self.Mel_Decoder = New_Inpainting_Networks.MelDecoder_old(hparams=hparams, norm_layer=hparams.normlayer)
        self.netD = Discriminator_Networks.MelDiscriminator()
        self.VideoEncoder = Image_Embedding.ImageEmbedding2(hparams=hparams)
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
            self.Mel_Decoder = torch.nn.DataParallel(self.Mel_Decoder)
            self.netD = torch.nn.DataParallel(self.netD)

            self.VideoEncoder = torch.nn.DataParallel(self.VideoEncoder)
            self.decode_model = torch.nn.DataParallel(self.decode_model)
        self.decode_model.to(device)
        self.Mel_Encoder.to(device)
        self.Mel_Decoder.to(device)
        self.netD.to(device)
        self.criterionL1.to(device)
        self.criterionGAN.to(device)

        self.VideoEncoder.to(device)

        ###########******** optimizer *******#########################
        if self.update_wavenet:
            param_groups = [{'params': self.decode_model.parameters(), 'lr_mult': 1.0},
                            {'params': self.Mel_Encoder.parameters(), 'lr_mult': 1.0},
                            {'params': self.Mel_Decoder.parameters(), 'lr_mult': 1.0}]
        else:
            param_groups = [{'params': self.decode_model.parameters(), 'lr_mult': 1.0},
                            {'params': self.Mel_Encoder.parameters(), 'lr_mult': 1.0}]
        self.optimizer_G = optim.Adam(param_groups,
                               lr=hparams.initial_learning_rate, betas=(
                                hparams.adam_beta1, hparams.adam_beta2),
                               eps=hparams.adam_eps, weight_decay=hparams.weight_decay,
                               amsgrad=hparams.amsgrad)

        ##############

        self.optimizer_D = torch.optim.Adam(list(self.netD.parameters()),
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

    def set_test_inputs(self, input):
        self.audio_c = input[2]
        self.audio_x_0 = input[3]
        self.audio_y_0 = input[4]
        self.image_block = input[0]
        self.flow_block = input[1]
        self.input_lengths_0 = input[6]
        self.audio_g = None
        self.input_path = input[7]
        self.mel_y = self.audio_c.view(self.audio_c.size(0), 1, self.hparams.cin_channels, -1)
        blank_length = 40
        self.blank_length = blank_length
        shape = self.mel_y.size()
        self.select_image_start = 25
        self.select_image_end = self.select_image_start + int(blank_length / 4)
        self.select_start = self.select_image_start * 4
        self.select_end = self.select_start + blank_length
        fill_content = torch.ones(shape[0], shape[1], shape[2], 2)
        fill_content[:, :, :, 0] = self.mel_y[:, :, :, self.select_start - 1].detach()
        fill_content[:, :, :, 1] = self.mel_y[:, :, :, self.select_end + 1].detach()
        fill_content = F.interpolate(fill_content, size=[shape[2], blank_length], mode='bilinear', align_corners=True)

        self.mel_x = self.mel_y.clone()
        self.mel_x[:, :, :, self.select_start: self.select_end] = fill_content
        self.mel_x_show = self.mel_x.clone().cpu()
        self.mel_x_show[:,:,:, self.select_start: self.select_end] = 0
        self.input_lengths = self.input_lengths_0[:self.hparams.wavenet_batch].clone()
        self.gen_start = self.select_start - 10
        self.gen_end = self.select_start + 50

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

        self.mel_y = self.mel_y.to(device)
        self.mel_x = self.mel_x.to(device)
        x_size = self.mel_y.size()
        net = self.Mel_Encoder.forward(self.mel_x)
        self.fake_mel = self.Mel_Decoder(net, x_size)

    def set_inputs(self, input):
        self.mel_y = input.view(input.size(0), 1, self.hparams.cin_channels, -1)

        self.create_mel_x()


    def create_mel_x(self):

        shape = self.mel_y.size()
        length = shape[3]
        blank_length = min(self.blank_length, int(length/4))
        if blank_length > 0:
            self.select_start = random.randint(0, length - blank_length - 1)

            self.select_end = self.select_start + blank_length

            assert self.select_end <= length - 1

            if self.select_start != 0 and self.select_end != length - 1:
                fill_content = torch.ones(shape[0], shape[1], shape[2], 2)
                fill_content[:, :, :, 0] = self.mel_y[:, :, :, self.select_start -1].detach()
                fill_content[:, :, :, 1] = self.mel_y[:, :, :, self.select_end + 1].detach()
                fill_content = F.interpolate(fill_content, size=[shape[2], blank_length], mode='bilinear', align_corners=True)
            elif self.select_start == 0:
                fill_content = self.mel_y[:,:,:, self.select_start + 1].detach()
                fill_content = fill_content.view(shape[0], shape[1], shape[2], 1)
                fill_content = F.interpolate(fill_content, size=[shape[2], blank_length], mode='bilinear', align_corners=True)
            else:
                fill_content = self.mel_y[:,:,:, self.select_start - 1].detach()
                fill_content = fill_content.view(shape[0], shape[1], shape[2], 1)
                fill_content = F.interpolate(fill_content, size=[shape[2], blank_length], mode='bilinear', align_corners=True)

            self.mel_x = self.mel_y.clone()
            self.mel_x[:,:,:, self.select_start: self.select_end] = fill_content

        else:
            self.mel_x = self.mel_y.clone()

        self.mel_x_show = self.mel_x.clone().cpu()
        if blank_length > 0:
            self.mel_x_show[:,:,:, self.select_start: self.select_end] = 0


    def forward(self):

        self.mel_y = self.mel_y.to(device)
        self.mel_x = self.mel_x.to(device)
        x_size = self.mel_y.size()
        net = self.Mel_Encoder.forward(self.mel_x)
        self.fake_mel = self.Mel_Decoder(net, x_size)

        self.loss_mel_L1 = self.criterionL1(self.fake_mel * 255, self.mel_y * 255)

    def backward_D(self):
        fake_AB = self.fake_mel
        # Real
        real_AB = self.mel_y # GroundTruth
        self.pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False, softlabel=True)
        self.pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True, softlabel=True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()


    def backward_G(self):


        pred_fake = self.netD(self.fake_mel)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True, softlabel=False)



        self.loss_G = self.loss_G_GAN + self.loss_mel_L1 * self.hparams.lambda_A

        self.loss_G.backward()

    def optimize_parameters(self, step):
        self.Mel_Encoder.train()
        self.Mel_Decoder.train()
        # Learning rate schedule
        if self.hparams.lr_schedule is not None:
            lr_schedule_f = getattr(lrschedule, self.hparams.lr_schedule)
            self.current_lr = lr_schedule_f(
                    self.hparams.initial_learning_rate, step, **self.hparams.lr_schedule_kwargs)
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = self.current_lr
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = self.current_lr

        self.forward()
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.get_loss_items()


        # update moving average

    def test(self):
        with torch.no_grad():
            self.forward()
        self.loss_mel_L1_item = self.loss_mel_L1.item()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN_item),
                            ('G_L1', self.loss_mel_L1_item),
                            ('D_real', self.loss_D_real_item),
                            ('D_fake', self.loss_D_fake_item)
                            ])

    def get_loss_items(self):
        self.loss_mel_L1_item = self.loss_mel_L1.item()
        self.loss_G_GAN_item = self.loss_G_GAN.item()
        self.loss_D_real_item = self.loss_D_real.item()
        self.loss_D_fake_item = self.loss_D_fake.item()

    def del_no_need(self):

        no_need_list = [self.mel_x, self.mel_y, self.loss_mel_L1, self.loss_G_GAN, self.loss_D_fake, self.loss_D_real]

        for values in no_need_list:
            del values

    def save_inpainting_checkpoint(self, global_step, global_test_step, checkpoint_dir, epoch, hparams=hparams):
        checkpoint_path = os.path.join(
                checkpoint_dir, hparams.name + "_checkpoint_step{:09d}.pth.tar".format(global_step))
        optimizer_G_state = self.optimizer_G.state_dict() if hparams.save_optimizer_state else None
        optimizer_D_state = self.optimizer_D.state_dict() if hparams.save_optimizer_state else None
        torch.save({
            "Mel_Encoder": self.Mel_Encoder.state_dict(),
            "Mel_Decoder": self.Mel_Decoder.state_dict(),
            "netD": self.netD.state_dict(),
            # "VideoEncoder": self.VideoEncoder.state_dict(),
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
        self.Mel_Decoder = util.copy_state_dict(checkpoint["Mel_Decoder"], self.Mel_Decoder)
        self.netD = util.copy_state_dict(checkpoint["netD"], self.netD)
        #self.VideoEncoder = util.copy_state_dict(checkpoint["VideoEncoder"], self.VideoEncoder)
        #self.Inpainting_Dis = util.copy_state_dict(checkpoint["Inpainting_Dis"], self.Inpainting_Dis)
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
        print("Load checkpoint from: {}".format(self.hparams.wavenet_pretrain))
        checkpoint = torch.load(self.hparams.wavenet_pretrain)
        self.decode_model = util.copy_state_dict(checkpoint["model"], self.decode_model)
