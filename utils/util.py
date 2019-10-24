from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
from sklearn.metrics.pairwise import euclidean_distances
import torch.nn as nn
import os
import shutil
import collections
import torch.nn.functional as F
import cv2
import Options_inpainting
hparams = Options_inpainting.Inpainting_Config()


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array

def to_np(x):
    return x.data.cpu().numpy()

def tensor2im(image_tensor, imtype=np.uint8, idx=0):
    image_numpy = image_tensor[idx].cpu().float().numpy()
    image_numpy = image_numpy * 255.0
    PIL_image = image_numpy

    return PIL_image.astype(imtype)

def tensor2image(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.expand(image_tensor.size(0), 3, image_tensor.size(2), image_tensor.size(3)).cpu()
    image_numpy = image_tensor[0].float().numpy()
    image_numpy = image_numpy * 255.0
    PIL_image = image_numpy

    return PIL_image.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def l2_norm(x):
    x_norm = F.normalize(x, p=2, dim=1)
    return x_norm


def L2retrieval(clips_embed, captions_embed, return_ranks = False):
    captions_num = captions_embed.shape[0]
    #index_list = []
    ranks = np.zeros(captions_num)
    top1 = np.zeros(captions_num)
    d = euclidean_distances(captions_embed, clips_embed)
    inds = np.argsort(d)
    num = np.arange(captions_num).reshape(captions_num, 1)
    ranks = np.where(inds == num)[1]
    top1 = inds[:, 0]
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    # r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
    #plus 1 because the index starts from 0
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, r50, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, r50, medr, meanr)


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, nn.Parameter):
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

def save_inpainting_checkpoint(model, global_step, global_test_step, checkpoint_dir, epoch, hparams=hparams):
    checkpoint_path = os.path.join(
        checkpoint_dir, hparams.name + "_checkpoint_step{:09d}.pth.tar".format(global_step))
    optimizer_G_state = model.optimizer_G.state_dict() if hparams.save_optimizer_state else None
    optimizer_D_state = model.optimizer_D.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "Mel_Encoder": model.Mel_Encoder.state_dict(),
        "Mel_Decoder": model.Mel_Decoder.state_dict(),
        "netD": model.netD.state_dict(),
        # "VideoEncoder": model.VideoEncoder.state_dict(),
        "optimizer_G": optimizer_G_state,
        "optimizer_D": optimizer_D_state,
        "global_step": global_step,
        "global_epoch": epoch,
        "global_test_step": global_test_step,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def load_part_checkpoint(path, model):

    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path)
    model.Mel_Encoder = copy_state_dict(checkpoint["Mel_Encoder"], model.Mel_Encoder)
    model.Mel_Decoder = copy_state_dict(checkpoint["Mel_Decoder"], model.Mel_Decoder)
    # model.netD = copy_state_dict(checkpoint["netD"], model.netD)

    return model
