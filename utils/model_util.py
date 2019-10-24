import Config

import os
from os.path import dirname, join, expanduser
from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw, is_scalar_input
import numpy as np
from warnings import warn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wavenet_vocoder import WaveNet

import torch
from torch.nn import functional as F
from wavenet_vocoder.mixture import sample_from_discretized_mix_logistic
from nnmnkwii import preprocessing as P

import librosa.display
hparams = Config.Config()
use_cuda = torch.cuda.is_available()


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def save_waveplot(path, y_hat, y_target):
    sr = hparams.sample_rate

    plt.figure(figsize=(16, 6))
    plt.subplot(2, 1, 1)
    librosa.display.waveplot(y_target, sr=sr)
    plt.subplot(2, 1, 2)
    librosa.display.waveplot(y_hat, sr=sr)
    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()


def save_states(global_step, y_hat, y, input_lengths, checkpoint_dir=None):
    print("Save intermediate states at step {}".format(global_step))
    idx = np.random.randint(0, len(y_hat))
    length = input_lengths[idx].data.cpu().item()

    # (B, C, T)
    if y_hat.dim() == 4:
        y_hat = y_hat.squeeze(-1)

    if is_mulaw_quantize(hparams.input_type):
        # (B, T)
        y_hat = F.softmax(y_hat, dim=1).max(1)[1]

        # (T,)
        y_hat = y_hat[idx].data.cpu().long().numpy()
        y = y[idx].view(-1).data.cpu().long().numpy()


        y_hat = y_hat.max(1)[1].view(-1).long().cpu().data.numpy()
        y_hat = P.inv_mulaw_quantize(y_hat, hparams.quantize_channels)
        y = P.inv_mulaw_quantize(y, hparams.quantize_channels)
    else:
        # (B, T)
        y_hat = sample_from_discretized_mix_logistic(
            y_hat, log_scale_min=hparams.log_scale_min)
        # (T,)
        y_hat = y_hat[idx].view(-1).data.cpu().numpy()
        y = y[idx].view(-1).data.cpu().numpy()

        if is_mulaw(hparams.input_type):
            y_hat = P.inv_mulaw(y_hat, hparams.quantize_channels)
            y = P.inv_mulaw(y, hparams.quantize_channels)

    # Mask by length
    y_hat[length:] = 0
    y[length:] = 0

    # Save audio
    audio_dir = join(checkpoint_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    path = join(audio_dir, "step{:09d}_predicted.wav".format(global_step))
    librosa.output.write_wav(path, y_hat, sr=hparams.sample_rate)
    path = join(audio_dir, "step{:09d}_target.wav".format(global_step))
    librosa.output.write_wav(path, y, sr=hparams.sample_rate)

    path = os.path.join(audio_dir, "step{:09d}_waveplots.png".format(global_step))
    save_waveplot(path, y_hat, y)

def restore_parts(state, model):

    model_dict = model.state_dict()
    valid_state_dict = {k: v for k, v in state.items() if k in model_dict}

    try:
        model_dict.update(valid_state_dict)
        model.load_state_dict(model_dict)
    except RuntimeError as e:
        # there should be invalid size of weight(s), so load them per parameter
        print(str(e))
        model_dict = model.state_dict()
        for k, v in valid_state_dict.items():
            model_dict[k] = v
            try:
                model.load_state_dict(model_dict)
            except RuntimeError as e:
                print(str(e))
                warn("{}: may contain invalid size of weight. skipping...".format(k))


def clone_as_averaged_model(averaged_model, model, ema):
    assert ema is not None
    averaged_model.load_state_dict(model.state_dict())
    for name, param in averaged_model.named_parameters():
        if name in ema.shadow:
            param.data = ema.shadow[name].clone()
    return averaged_model


def save_checkpoint(device, model, global_step, global_test_step, checkpoint_dir, epoch, ema=None):
    checkpoint_path = join(
        checkpoint_dir, hparams.name + "_checkpoint_step{:09d}.pth.tar".format(global_step))
    optimizer_state = model.optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "model": model.decode_model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": global_step,
        "global_epoch": epoch,
        "global_test_step": global_test_step,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

    if ema is not None:
        averaged_model = WaveNet(scalar_input=is_scalar_input(hparams.input_type))
        averaged_model = torch.nn.DataParallel(averaged_model).to(device)
        averaged_model = clone_as_averaged_model(averaged_model, model, ema)
        checkpoint_path = join(
            checkpoint_dir, "checkpoint_step{:09d}_ema.pth".format(global_step))
        torch.save({
            "model": averaged_model.state_dict(),
            "optimizer": optimizer_state,
            "global_step": global_step,
            "global_epoch": epoch,
            "global_test_step": global_test_step,
        }, checkpoint_path)
        print("Saved averaged checkpoint:", checkpoint_path)


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model


def load_pretrain_checkpoint(path, model):

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model = copy_state_dict(checkpoint['state_dict'], model)

    return model
