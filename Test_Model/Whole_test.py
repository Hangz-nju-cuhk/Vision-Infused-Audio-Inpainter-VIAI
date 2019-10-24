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


class AudioModel():

    def __init__(self, hparams, device=device):
        self.Tensor = torch.Tensor
        self.train = 1
        self.use_gan = 0
        self.device = device
        self.hparams = hparams
        self.name = hparams.name
        self.fs = hparams.sample_rate
        self.update_wavenet = False
        self.decode_model = WaveNet(scalar_input=is_scalar_input(hparams.input_type))
        self.Mel_Encoder = Inpainting_Networks.MelEncoder(hparams=hparams)
        self.Mel_Decoder_Image = New_Inpainting_Networks.MelDecoderImage2(hparams=hparams, norm_layer=hparams.normlayer)
        self.Mel_Decoder = New_Inpainting_Networks.MelDecoder_old(hparams=hparams, norm_layer=hparams.normlayer)
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
            self.Mel_Decoder = torch.nn.DataParallel(self.Mel_Decoder)
            self.Inpainting_Dis = torch.nn.DataParallel(self.Inpainting_Dis)
            # self.ResNet = torch.nn.DataParallel(self.ResNet)
            self.decode_model = torch.nn.DataParallel(self.decode_model)
            # self.ImageEmbedding_finetune = torch.nn.DataParallel(self.ImageEmbedding_finetune)
        self.decode_model.to(device)
        self.Mel_Encoder.to(device)
        self.netD.to(device)
        self.VideoEncoder.to(device)
        self.Mel_Decoder_Image.to(device)
        self.Mel_Decoder.to(device)
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
        self.input_lengths = self.input_lengths_0.clone()
        self.input_lengths = self.input_lengths.fill_((self.select_end + 1) * 320)

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
        mel_net = self.Mel_Encoder.forward(self.mel_y)

        inpainting_net = self.Mel_Encoder.forward(self.mel_x)

        x_size = self.mel_y.size()

        self.video_net, self.vid_fea = self.VideoEncoder.forward(self.image_block, self.flow_block)
        # self.video_net = self.VideoEncoder.forward(self.image_block)

        self.audio_net = mel_net[-1].detach()

        self.fake_mel = self.Mel_Decoder_Image(inpainting_net, x_size, self.video_net)

        no_need_list = [self.video_net, self.image_block, self.flow_block, self.VideoEncoder, self.Mel_Decoder_Image]

        for values in no_need_list:
            del values


    def set_inputs(self, input):

        # video_block, flow_block, mel_block, label = input
        self.audio_c = input[2]
        self.audio_x_0 = input[3]
        self.audio_y_0 = input[4]
        self.image_block = input[0]
        self.flow_block = input[1]
        self.input_lengths_0 = input[6]
        self.audio_g = None
        self.input_path = input[7]

        batch_size = self.audio_c.size(0)
        random_batch = np.arange(batch_size)
        random.shuffle(random_batch)
        self.image_block = self.image_block[random_batch]
        self.flow_block = self.flow_block[random_batch]
        self.audio_c = self.audio_c[random_batch]
        self.audio_x_0 = self.audio_x_0[random_batch]
        self.audio_y_0 = self.audio_y_0[random_batch]
        self.input_path = list(np.array(self.input_path)[random_batch])
        self.mel_y = self.audio_c.view(self.audio_c.size(0), 1, self.hparams.cin_channels, -1)

        self.create_mel_x_2()
        ###########********* crop ****************######################

        self.audio_x = self.audio_x_0[:self.hparams.wavenet_batch, :, self.gen_start * 320:self.gen_end * 320].clone()
        self.audio_y = self.audio_y_0[:self.hparams.wavenet_batch, self.gen_start * 320:self.gen_end * 320, :].clone()
        self.input_lengths = self.input_lengths_0[:self.hparams.wavenet_batch].clone()
        self.input_lengths = self.input_lengths.fill_(self.gen_end * 320 - self.gen_start * 320)


    def create_mel_x_2(self):

        shape = self.mel_y.size()
        length = shape[3]
        blank_length = min(self.blank_length, int(length/4))
        if blank_length > 0:
            self.select_image_start = random.randint(int(length/8) - 5, int(length/8) + 2)

            self.select_image_end = self.select_image_start + int(blank_length/4)

            self.select_start = self.select_image_start * 4

            self.select_end = self.select_start + blank_length


            assert self.select_end <= length - 1

            fill_content = torch.ones(shape[0], shape[1], shape[2], 2)
            fill_content[:, :, :, 0] = self.mel_y[:, :, :, self.select_start -1].detach()
            fill_content[:, :, :, 1] = self.mel_y[:, :, :, self.select_end + 1].detach()
            fill_content = F.interpolate(fill_content, size=[shape[2], blank_length], mode='bilinear', align_corners=True)

            self.mel_x = self.mel_y.clone()
            self.mel_x[:,:,:, self.select_start: self.select_end] = fill_content

        else:
            self.mel_x = self.mel_y.clone()

        self.mel_x_show = self.mel_x.clone().cpu()
        if blank_length > 0:
            self.mel_x_show[:,:,:, self.select_start: self.select_end] = 0

        self.gen_start = self.select_start - 10
        self.gen_end = self.select_start + 50


    def forward_inpainting(self):

        self.mel_y = self.mel_y.to(device)
        self.mel_x = self.mel_x.to(device)
        self.image_block = self.image_block.to(device)
        self.flow_block = self.flow_block.to(device)
        self.mask = torch.zeros_like(self.mel_x)
        self.mask[:,:,:, self.select_start: self.select_end] = 1
        self.mask = self.mask.to(device)
        mel_net = self.Mel_Encoder.forward(self.mel_y)

        inpainting_net = self.Mel_Encoder.forward(self.mel_x)

        x_size = self.mel_y.size()

        self.video_net, self.vid_fea = self.VideoEncoder.forward(self.image_block, self.flow_block)
        # self.video_net = self.VideoEncoder.forward(self.image_block)


        self.audio_net = mel_net[-1].detach()

        self.matching_L1 = self.criterionL1(self.video_net, self.audio_net) * self.hparams.lambda_B

        self.fake_mel = self.Mel_Decoder_Image(inpainting_net, x_size, self.video_net)


        self.mel_net_norm = util.l2_norm(self.audio_net.view(-1, self.hparams.length_feature * 13))
        self.video_net_norm = util.l2_norm(self.video_net.view(-1, self.hparams.length_feature * 13))

        self.mel_net_norm_buffer = self.mel_net_norm.detach()
        self.video_net_norm_buffer = self.video_net_norm.detach()

        self.EmbeddingL2_video = self.L2Contrastive.forward(self.video_net_norm, self.mel_net_norm_buffer)
        self.EmbeddingL2_audio = self.L2Contrastive.forward(self.mel_net_norm, self.video_net_norm_buffer)
        self.loss_mel_L1 = self.criterionL1(self.fake_mel * 255, self.mel_y * 255) * 0.001 + \
                           self.criterionL1(self.fake_mel * 255 * self.mask, self.mel_y * 255 * self.mask) * 5

        self.fake_audio_c = self.fake_mel.view(self.audio_c.size(0), self.audio_c.size(1), self.audio_c.size(2))[:self.hparams.wavenet_batch, :,
                        self.gen_start:self.gen_end]



    def forward(self):

        self.audio_x, self.audio_y = self.audio_x.to(self.device), \
                                     self.audio_y.to(self.device)
        self.input_lengths = self.input_lengths.to(self.device)
        self.forward_inpainting()

        # self.audio_c = self.audio_c.to(self.device) if self.audio_c is not None else None
        self.audio_g = self.audio_g.to(self.device) if self.audio_g is not None else None
        # self.y_hat = torch.nn.parallel.data_parallel(self.decode_model, (self.audio_x, self.audio_c, self.audio_g, False))
        if self.update_wavenet:
            self.y_hat = self.decode_model.forward(self.audio_x, self.fake_audio_c, self.audio_g, False)
            mask = self.sequence_mask(self.input_lengths, max_len=self.audio_x.size(-1)).unsqueeze(-1)
            mask = mask[:, 1:, :]
            self.audio_L2loss = 0
            self.L2loss = 0
            if is_mulaw_quantize(hparams.input_type):
                # wee need 4d inputs for spatial cross entropy loss
                # (B, C, T, 1)
                self.y_hat = self.y_hat.unsqueeze(-1)
                self.audio_reconstruct_loss = self.audio_criterion(self.y_hat[:, :, :-1, :], self.audio_y[:, 1:, :],
                                                                  mask=mask)
                self.y_hat = self.y_hat.squeeze(-1)
            else:
                self.audio_reconstruct_loss = self.audio_criterion(self.y_hat[:, :, :-1], self.audio_y[:, 1:, :],
                                                                  mask=mask)
                if hparams.alpha > 0:
                    x = sample_from_discretized_mix_logistic(
                            self.y_hat, log_scale_min=hparams.log_scale_min)
                    # B x C x T
            self.reconstruct_loss_item = self.audio_reconstruct_loss.item()
        else:
            self.audio_reconstruct_loss = 0
            self.recontruct_loss = 0

    def backward_dis(self):
        self.audio_D_real = self.discriminator_audio(self.audio_net.detach())
        self.audio_D_fake = self.discriminator_audio( self.video_net.detach())
        self.loss_mel_D_fake = self.criterionGAN(self.audio_D_fake, False, softlabel=True)
        self.loss_mel_D_real = self.criterionGAN(self.audio_D_real, True, softlabel=True)
        self.dis_R_loss = (self.loss_mel_D_fake + self.loss_mel_D_real) * 0.5
        self.dis_R_loss.backward()

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

    def backward_D2(self):
        fake_AB = self.fake_mel
        # Real
        real_AB = self.mel_y # GroundTruth
        self.pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False, softlabel=True)
        self.pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True, softlabel=True)
        self.loss_D_single = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.fake_inpainting = self.fake_mel[:,:,:, self.select_start: self.select_start + 48]
        self.feature_part = self.vid_fea[:,:, self.select_image_start: self.select_image_start + 12]
        self.true_inpainting = self.mel_y[:,:,:, self.select_start: self.select_start + 48]

        self.pred_fake_combine = self.Inpainting_Dis(self.fake_inpainting.detach(), self.feature_part.detach())
        self.loss_D_fake_combine = self.criterionGAN(self.pred_fake_combine, False, softlabel=True)
        self.pred_real_combine = self.Inpainting_Dis(self.true_inpainting, self.feature_part)
        self.loss_D_real_combine = self.criterionGAN(self.pred_real, True, softlabel=True)
        self.loss_D_combine = (self.loss_D_fake_combine + self.loss_D_real_combine) * 0.5

        self.loss_D = self.loss_D_single + self.loss_D_combine

        self.loss_D.backward()

    def backward_G(self):

        pred_fake_combine = self.Inpainting_Dis(self.fake_inpainting, self.feature_part)
        self.loss_G_GAN_combine = self.criterionGAN(pred_fake_combine, True, softlabel=False)

        pred_fake = self.netD(self.fake_mel)
        #
        self.loss_G_GAN = self.criterionGAN(pred_fake, True, softlabel=False)

        # self.loss_G = self.loss_G_GAN + self.loss_mel_L1 * self.hparams.lambda_A
        if self.use_gan:
            audio_D_fake = self.discriminator_audio(self.video_net)

            self.image_loss_D_inv = self.criterionGAN(audio_D_fake, True)

            self.loss_G = self.EmbeddingL2_video + self.EmbeddingL2_audio + self.matching_L1 + self.loss_mel_L1 + self.image_loss_D_inv

        else:

            self.loss_G = self.EmbeddingL2_video + self.EmbeddingL2_audio + self.matching_L1 * 0.1 +\
                          self.loss_mel_L1 * self.hparams.lambda_A + self.loss_G_GAN + self.loss_G_GAN_combine

        self.loss_G.backward()

    def optimize_parameters(self, step):
        self.decode_model.train()
        self.Mel_Encoder.train()
        self.VideoEncoder.train()
        self.Mel_Decoder_Image.train()
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
        if self.use_gan:
            self.backward_dis()
        self.backward_D2()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        if self.clip_thresh > 0:
            self.grad_norm = torch.nn.utils.clip_grad_norm_(self.decode_model.parameters(), self.clip_thresh)
        self.optimizer_G.step()

        self.get_loss_items()
        if self.ema is not None:
            for name, param in self.decode_model.named_parameters():
                if name in self.ema.shadow:
                    self.ema.update(name, param.data)

        # update moving average

    def test(self):
        with torch.no_grad():
            self.Mel_Encoder.eval()
            self.Mel_Decoder_Image.eval()
            self.VideoEncoder.eval()
            self.forward()
            self.get_loss_items()
            self.del_no_need()

    def get_loss_items(self):
        self.loss_mel_L1_item = self.loss_mel_L1.item()
        if self.train:
            self.loss_G_GAN_item = self.loss_G_GAN.item()
            self.loss_D_real_item = self.loss_D_real.item()
            self.loss_D_fake_item = self.loss_D_fake.item()
            self.loss_G_GAN_combine_item = self.loss_G_GAN_combine.item()
            self.loss_D_real_combine_item = self.loss_D_real_combine.item()
            self.loss_D_fake_combine_item = self.loss_D_fake_combine.item()
            if self.use_gan:
                self.image_loss_D_inv_item = self.image_loss_D_inv.item()
                self.loss_mel_D_real_item = self.loss_mel_D_real.item()
                self.loss_mel_D_fake_item = self.loss_mel_D_fake.item()
        self.matching_L1_item = self.matching_L1.item()
        self.EmbeddingL2_item = self.EmbeddingL2_video.item()
        self.EmbeddingL2_audio_item = self.EmbeddingL2_audio.item()


    def eval_model(self, global_step, eval_dir):
        if self.ema is not None:
            print(self.name + "Using averaged model for evaluation")
            average_model = WaveNet(scalar_input=is_scalar_input(self.hparams.input_type))
            if self.model_parallel:
                average_model = torch.nn.DataParallel(average_model)
            average_model = average_model.to(self.device)
            model = model_util.clone_as_averaged_model(average_model, self.decode_model, self.ema)
            if self.model_parallel:
                model.module.make_generation_fast_()
            else:
                model.make_generation_fast_()
        else:
            model = self.decode_model

        model.eval()
        idx = np.random.randint(0, len(self.audio_y))
        length = self.input_lengths[idx].data.cpu().item()
        y_target = self.audio_y[idx].view(-1).detach().cpu().numpy()[:length]
        with torch.no_grad():
            # self.VideoEncoder.eval()
            # self.Mel_Decoder_Image.eval()
            # self.Mel_Encoder.eval()
            self.forward_inpainting()
        if self.audio_c is not None:
            c = self.fake_audio_c
            c = c[idx, :, :length].unsqueeze(0)
            assert c.dim() == 3
            print("Shape of local conditioning features: {}".format(c.size()))
        else:
            c = None
        if self.audio_g is not None:
            # TODO: test
            g = self.audio_g[idx]
            print("Shape of global conditioning features: {}".format(g.size()))
        else:
            g = None
        # Dummy silence
        if is_mulaw_quantize(self.hparams.input_type):
            initial_value = P.mulaw_quantize(0, self.hparams.quantize_channels)
        elif is_mulaw(self.hparams.input_type):
            initial_value = P.mulaw(0.0, self.hparams.quantize_channels)
        else:
            initial_value = 0.0
        print("Intial value:", initial_value)

        if is_mulaw_quantize(hparams.input_type):
            initial_input = np_utils.to_categorical(
                    initial_value, num_classes=self.hparams.quantize_channels).astype(np.float32)
            initial_input = torch.from_numpy(initial_input).view(
                    1, 1, self.hparams.quantize_channels)
        else:
            initial_input = torch.zeros(1, 1, 1).fill_(initial_value)
        initial_input = initial_input.to(self.device)

        with torch.no_grad():
            if self.model_parallel:
                y_hat = model.module.incremental_forward(
                        initial_input, c=c, g=g, T=length, softmax=True, quantize=True, tqdm=tqdm,
                        log_scale_min=self.hparams.log_scale_min)
            else:
                y_hat = model.incremental_forward(
                        initial_input, c=c, g=g, T=length, softmax=True, quantize=True, tqdm=tqdm,
                        log_scale_min=self.hparams.log_scale_min)


        if is_mulaw_quantize(self.hparams.input_type):
            y_hat = y_hat.max(1)[1].view(-1).long().cpu().data.numpy()
            y_hat = P.inv_mulaw_quantize(y_hat, self.hparams.quantize_channels)
            y_target = P.inv_mulaw_quantize(y_target, self.hparams.quantize_channels)
        elif is_mulaw(self.hparams.input_type):
            y_hat = P.inv_mulaw(y_hat.view(-1).cpu().data.numpy(), self.hparams.quantize_channels)
            y_target = P.inv_mulaw(y_target, self.hparams.quantize_channels)
        else:
            y_hat = y_hat.view(-1).cpu().data.numpy()
        # Save audio
        audio_dir = os.path.join(eval_dir, hparams.name + "_audio")
        os.makedirs(audio_dir, exist_ok=True)
        path = os.path.join(audio_dir, hparams.name + "_step{:09d}_predicted.wav".format(global_step))
        librosa.output.write_wav(path, y_hat, sr=self.hparams.sample_rate)
        path = os.path.join(audio_dir, hparams.name + "_step{:09d}_target.wav".format(global_step))
        librosa.output.write_wav(path, y_target, sr=self.hparams.sample_rate)
        y_input = y_target.copy()
        y_input[10 * 320 : (10 + self.blank_length) * 320] = 0
        path = os.path.join(audio_dir, hparams.name + "_step{:09d}_input.wav".format(global_step))
        librosa.output.write_wav(path, y_input, sr=self.hparams.sample_rate)
        # save figure
        path = os.path.join(audio_dir, hparams.name + "_step{:09d}_waveplots.png".format(global_step))
        model_util.save_waveplot(path, y_hat, y_target)

        no_need_list = [model, y_hat]
        for item in no_need_list:
            del item


    def eval_model_test(self, global_step, eval_dir, idx=None):
        if self.ema is not None:
            print(self.name + " Using averaged model for evaluation")
            average_model = WaveNet(scalar_input=is_scalar_input(self.hparams.input_type))
            if self.model_parallel:
                average_model = torch.nn.DataParallel(average_model)
            average_model = average_model.to(self.device)
            model = model_util.clone_as_averaged_model(average_model, self.decode_model, self.ema)
            if self.model_parallel:
                model.module.make_generation_fast_()
            else:
                model.make_generation_fast_()
        else:
            model = self.decode_model

        model.eval()
        if idx is None:
            idx = np.random.randint(0, len(self.audio_y_0))

        length = self.input_lengths_0[idx].data.cpu().item()
        y_target = self.audio_y_0[idx].view(-1).detach().cpu().numpy()[:length]
        # set test_inputs
        test_inputs = self.audio_y_0[idx, :self.select_start * self.hparams.hop_size].to(self.device)
        test_inputs = test_inputs.unsqueeze(0)
        with torch.no_grad():
            # self.VideoEncoder.eval()
            # self.Mel_Decoder_Image.eval()
            # self.Mel_Encoder.eval()
            # self.Mel_Decoder.eval()
            self.test_inpainting()
        if self.audio_c is not None:
            c = self.mask * self.fake_mel + self.inverse_mask * self.mel_y
            c = c.view(self.audio_c.size(0), self.audio_c.size(1), self.audio_c.size(2))
            c = c[idx, :, :int(length/self.hparams.hop_size)].unsqueeze(0)


            assert c.dim() == 3
            print("Shape of local conditioning features: {}".format(c.size()))
        else:
            c = None
        if self.audio_g is not None:
            # TODO: test
            g = self.audio_g[idx]
            print("Shape of global conditioning features: {}".format(g.size()))
        else:
            g = None
        # Dummy silence
        if is_mulaw_quantize(self.hparams.input_type):
            initial_value = P.mulaw_quantize(0, self.hparams.quantize_channels)
        elif is_mulaw(self.hparams.input_type):
            initial_value = P.mulaw(0.0, self.hparams.quantize_channels)
        else:
            initial_value = 0.0
        print("Intial value:", initial_value)

        if is_mulaw_quantize(self.hparams.input_type):
            initial_input = np_utils.to_categorical(
                    initial_value, num_classes=self.hparams.quantize_channels).astype(np.float32)
            initial_input = torch.from_numpy(initial_input).view(
                    1, 1, self.hparams.quantize_channels)
        else:
            initial_input = torch.zeros(1, 1, 1).fill_(initial_value)
        initial_input = initial_input.to(self.device)
        audio_dir = eval_dir
        # audio_dir = os.path.join(eval_dir, self.name + "_audio")
        os.makedirs(audio_dir, exist_ok=True)
        visuals = self.get_current_visuals(idx)
        name = self.name + "_test_step{:09d}_mel.jpg".format(global_step)
        for label, image_numpy in visuals.items():
            image_numpy = image_numpy.squeeze(0)
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(audio_dir, image_name)
            util.save_image(image_numpy, save_path)

        numpys = self.get_current_mels(idx)
        num_name = self.name + "_test_step{:09d}_mel".format(global_step)
        for label, image_numpy in numpys.items():
            image_numpy = image_numpy.squeeze(0)
            image_name = '%s_%s.npy' % (num_name, label)
            save_path = os.path.join(audio_dir, image_name)
            np.save(save_path, image_numpy)

        with torch.no_grad():
            if self.model_parallel:
                y_hat = model.module.incremental_forward(
                        initial_input, c=c, g=g, T=length, test_inputs=test_inputs,  softmax=True, quantize=True, tqdm=tqdm,
                        log_scale_min=self.hparams.log_scale_min)
            else:
                y_hat = model.incremental_forward(
                        initial_input, c=c, g=g, T=length, test_inputs=test_inputs, softmax=True, quantize=True, tqdm=tqdm,
                        log_scale_min=self.hparams.log_scale_min)


        if is_mulaw_quantize(self.hparams.input_type):
            y_hat = y_hat.max(1)[1].view(-1).long().cpu().data.numpy()
            y_hat = P.inv_mulaw_quantize(y_hat, self.hparams.quantize_channels)
            y_target = P.inv_mulaw_quantize(y_target, self.hparams.quantize_channels)
        elif is_mulaw(self.hparams.input_type):
            y_hat = P.inv_mulaw(y_hat.view(-1).cpu().data.numpy(), self.hparams.quantize_channels)
            y_target = P.inv_mulaw(y_target, self.hparams.quantize_channels)
        else:
            y_hat = y_hat.view(-1).cpu().data.numpy()
        # Save audio
        y_hat_2 = y_target.copy()
        y_hat_2[self.select_start * 320: (self.select_start + self.blank_length)*320] = \
            y_hat[self.select_start * 320: (self.select_start + self.blank_length)*320]

        path = os.path.join(audio_dir, self.name + "_test_step{:09d}_predicted.wav".format(global_step))
        librosa.output.write_wav(path, y_hat_2, sr=self.hparams.sample_rate)
        path = os.path.join(audio_dir, self.name + "_test_step{:09d}_target.wav".format(global_step))
        librosa.output.write_wav(path, y_target, sr=self.hparams.sample_rate)
        y_input = y_target.copy()
        y_input[self.select_start * 320: (self.select_start + self.blank_length)*320] = 0
        path = os.path.join(audio_dir, self.name + "_test_step{:09d}_input.wav".format(global_step))
        librosa.output.write_wav(path, y_input, sr=self.hparams.sample_rate)
        path = os.path.join(audio_dir, self.name + "_test_step{:09d}_clip_inpainted.wav".format(global_step))
        librosa.output.write_wav(path, y_hat[self.select_start * 320: (self.select_end)*320], sr=self.hparams.sample_rate)
        path = os.path.join(audio_dir, self.name + "_test_step{:09d}_clip_original.wav".format(global_step))
        librosa.output.write_wav(path, y_target[self.select_start * 320: (self.select_end)*320], sr=self.hparams.sample_rate)
        print(self.input_path[idx])
        path = os.path.join(audio_dir, self.name + "_test_step{:09d}_waveplots.png".format(global_step))
        model_util.save_waveplot(path, y_hat_2, y_target)
        path = os.path.join(audio_dir, self.name + "_test_step{:09d}_generate_all.wav".format(global_step))
        librosa.output.write_wav(path, y_hat, sr=self.hparams.sample_rate)
        y_hat_2 = y_target.copy()
        y_hat_2[self.select_start * 320:] = \
            y_hat[self.select_start * 320:]
        path = os.path.join(audio_dir, self.name + "_test_step{:09d}_half.wav".format(global_step))
        librosa.output.write_wav(path, y_hat_2, sr=self.hparams.sample_rate)
        # save figure



        no_need_list = [model, y_hat_2, self.Mel_Encoder, self.fake_mel]
        for item in no_need_list:
            del item



    def model_test_interpolate(self, global_step, eval_dir, idx=None):
        if self.ema is not None:
            print(self.name + " Using averaged model for evaluation")
            average_model = WaveNet(scalar_input=is_scalar_input(self.hparams.input_type))
            if self.model_parallel:
                average_model = torch.nn.DataParallel(average_model)
            average_model = average_model.to(self.device)
            model = model_util.clone_as_averaged_model(average_model, self.decode_model, self.ema)
            if self.model_parallel:
                model.module.make_generation_fast_()
            else:
                model.make_generation_fast_()
        else:
            model = self.decode_model

        model.eval()
        if idx is None:
            idx = np.random.randint(0, len(self.audio_y_0))
        with torch.no_grad():
            # self.VideoEncoder.eval()
            # self.Mel_Decoder_Image.eval()
            # self.Mel_Encoder.eval()
            self.test_inpainting()
        length = self.input_lengths[idx].data.cpu().item()
        length_0 = self.input_lengths_0[idx].data.cpu().item()
        y_target = self.audio_y_0[idx].view(-1).detach().cpu().numpy()[:length_0]
        # set test_inputs
        if self.audio_c is not None:
            c = self.mel_x[:, :, :, : self.select_end + 1]
            c = c.view(c.size(0), c.size(2), c.size(3))
            c = c[idx].unsqueeze(0)
            assert c.dim() == 3
            print("Shape of local conditioning features: {}".format(c.size()))
        else:
            c = None
        # Dummy silence
        initial_value = 0.0
        print("Intial value:", initial_value)

        initial_input = torch.zeros(1, 1, 1).fill_(initial_value)
        initial_input = initial_input.to(self.device)
        # audio_dir = os.path.join(eval_dir, self.name + "_audio")
        audio_dir = eval_dir
        with torch.no_grad():
            if self.model_parallel:
                y_hat = model.module.incremental_forward(
                        initial_input, c=c, g=None, T=length, softmax=True, quantize=True, tqdm=tqdm,
                        log_scale_min=self.hparams.log_scale_min)
            else:
                y_hat = model.incremental_forward(
                        initial_input, c=c, g=None, T=length, softmax=True, quantize=True, tqdm=tqdm,
                        log_scale_min=self.hparams.log_scale_min)


        y_hat = y_hat.view(-1).cpu().data.numpy()
        # Save audio
        y_hat_2 = y_target.copy()
        y_hat_2[self.select_start * 320: (self.select_start + self.blank_length)*320] = \
            y_hat[: (self.blank_length)*320]
        path = os.path.join(audio_dir, self.name + "_test_step{:09d}_direct.wav".format(global_step))
        librosa.output.write_wav(path, y_hat_2, sr=self.hparams.sample_rate)
        ########################################################## interpolate ###################################################################
        if self.audio_c is not None:
            c = self.mel_x[:, :, :, self.select_start: self.select_end + 1]
            c = c.view(c.size(0), c.size(2), c.size(3))
            c = c[idx].unsqueeze(0)

            assert c.dim() == 3
            print("Shape of local conditioning features: {}".format(c.size()))
        else:
            c = None
        initial_value = 0.0
        print("Intial value:", initial_value)
        length =  (self.select_end + 1 - self.select_start) * 320
        initial_input = torch.zeros(1, 1, 1).fill_(initial_value)
        initial_input = initial_input.to(self.device)
        with torch.no_grad():
            if self.model_parallel:
                y_hat = model.module.incremental_forward(
                        initial_input, c=c, g=None, T=length, softmax=True, quantize=True, tqdm=tqdm,
                        log_scale_min=self.hparams.log_scale_min)
            else:
                y_hat = model.incremental_forward(
                        initial_input, c=c, g=None, T=length, softmax=True, quantize=True, tqdm=tqdm,
                        log_scale_min=self.hparams.log_scale_min)


        y_hat = y_hat.view(-1).cpu().data.numpy()
        # Save audio
        y_hat_2 = y_target.copy()
        y_hat_2[self.select_start * 320: (self.select_start + self.blank_length)*320] = \
            y_hat[: (self.blank_length)*320]

        path = os.path.join(audio_dir, self.name + "_test_step{:09d}_interpolate.wav".format(global_step))
        librosa.output.write_wav(path, y_hat_2, sr=self.hparams.sample_rate)

        no_need_list = [model, y_hat_2, self.Mel_Encoder, self.fake_mel]
        for item in no_need_list:
            del item


    def get_blank_space_length(self, global_epoch):
        self.blank_length = min( self.hparams.initial_blank_length + int(global_epoch ** self.hparams.blank_moving_rate - 1), int(self.hparams.max_mel_lengths / 4))

    def del_no_need(self):

        # no_need_list = [self.mel_x, self.mel_y, self.loss_mel_L1, self.loss_G_GAN, self.loss_D_fake, self.loss_D_real]
        no_need_list = [self.mel_x, self.mel_y, self.matching_L1, self.EmbeddingL2_video, self.EmbeddingL2_audio,
                        self.loss_mel_L1, self.video_net, self.image_block, self.flow_block, self.fake_mel]

        for values in no_need_list:
            del values

    def sequence_mask(self, sequence_length, max_len=None):
        if max_len is None:
            max_len = sequence_length.data.max()
        batch_size = sequence_length.size(0)
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = sequence_length.unsqueeze(1) \
            .expand_as(seq_range_expand)
        return (seq_range_expand < seq_length_expand).float()

    def get_current_errors(self):
        return OrderedDict([
                            # ('G_GAN', self.loss_G_GAN_item),
                            ('G_GAN', self.loss_G_GAN_item),
                            ('G_L1', self.loss_mel_L1_item),
                            ('D_real', self.loss_D_real_item),
                            ('D_fake', self.loss_D_fake_item),
                            ('loss_D_real_combine', self.loss_D_real_combine_item),
                            ('loss_G_GAN_combine', self.loss_G_GAN_combine_item),
                            ('matching_L1', self.matching_L1_item),
                            ('EmbeddingL2', self.EmbeddingL2_item)
                            ])

    def tensor2num(self, image_tensor, idx=0):
        return  image_tensor[idx].cpu().float().numpy()

    def get_current_visuals(self, idx=0):
        real_A = util.tensor2im(self.mel_x_show.data, idx=idx)
        fake_B = util.tensor2im(self.fake_mel.data, idx=idx)
        real_inter = util.tensor2im(self.mel_x.data, idx=idx)
        real_B = util.tensor2im(self.mel_y.data, idx=idx)
        return OrderedDict([('real_A', real_A), ('real_inter', real_inter), ('fake_B', fake_B), ('real_B', real_B)])

    def get_current_mels(self, idx=0):
        real_A = self.tensor2num(self.mel_x_show.data, idx=idx)
        fake_B = self.tensor2num(self.fake_mel.data, idx=idx)
        real_B = self.tensor2num(self.mel_y.data, idx=idx)
        return OrderedDict([('real_A', real_A),  ('fake_B', fake_B), ('real_B', real_B)])

    def TF_writer(self, writer, step):

        writer.add_scalar("L1loss", float(self.loss_mel_L1_item), step)
        writer.add_scalar("matching_L1", float(self.matching_L1_item), step)
        writer.add_scalar("EmbeddingL2", float(self.EmbeddingL2_item), step)

        if self.train:
            writer.add_scalar("G_GAN", float(self.loss_G_GAN_item), step)
            writer.add_scalar("D_real", float(self.loss_D_real_item), step)
            writer.add_scalar("D_fake", float(self.loss_D_fake_item), step)
            writer.add_scalar("learning rate", self.current_lr, step)
            if self.use_gan:
                writer.add_scalar("image_loss_D_inv", float(self.image_loss_D_inv_item), step)
                writer.add_scalar("mel_D_real", float(self.loss_mel_D_real_item), step)
                writer.add_scalar("mel_D_fake", float(self.loss_mel_D_fake_item), step)


    def save_inpainting_checkpoint(self, global_step, global_test_step, checkpoint_dir, epoch, hparams=hparams):
        checkpoint_path = os.path.join(
                checkpoint_dir, self.name + "_checkpoint_step{:09d}.pth.tar".format(global_step))
        optimizer_G_state = self.optimizer_G.state_dict() if hparams.save_optimizer_state else None
        optimizer_D_state = self.optimizer_D.state_dict() if hparams.save_optimizer_state else None
        torch.save({
            "Mel_Encoder": self.Mel_Encoder.state_dict(),
            "decode_model": self.decode_model.state_dict(),
            "Mel_Decoder_Image": self.Mel_Decoder_Image.state_dict(),
            "netD": self.netD.state_dict(),
            "Inpainting_Dis": self.Inpainting_Dis.state_dict(),
            "VideoEncoder": self.VideoEncoder.state_dict(),
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
        self.Mel_Decoder_Image = util.copy_state_dict(checkpoint["Mel_Decoder_Image"], self.Mel_Decoder_Image)
        self.netD = util.copy_state_dict(checkpoint["netD"], self.netD)
        self.VideoEncoder = util.copy_state_dict(checkpoint["VideoEncoder"], self.VideoEncoder)
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