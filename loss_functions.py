import Config
import torch
import torch.nn as nn
from wavenet_vocoder.mixture import discretized_mix_logistic_loss
from wavenet_vocoder.mixture import sample_from_discretized_mix_logistic
import random
import torch.nn.functional as F
hparams = Config.Config()


def sequence_mask(sequence_length, max_len=None):
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


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduce=False)

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, T, 1)
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)

        # (B, T, D)
        mask_ = mask.expand_as(target)
        losses = self.criterion(input, target)
        return ((losses * mask_).sum()) / mask_.sum()


class DiscretizedMixturelogisticLoss(nn.Module):
    def __init__(self):
        super(DiscretizedMixturelogisticLoss, self).__init__()

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, T, 1)
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)

        # (B, T, 1)
        mask_ = mask.expand_as(target)

        losses = discretized_mix_logistic_loss(
            input, target, num_classes=hparams.quantize_channels,
            log_scale_min=hparams.log_scale_min, reduce=False)
        assert losses.size() == target.size()
        return ((losses * mask_).sum()) / mask_.sum()


class ExponentialMovingAverage(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update(self, name, x):
        assert name in self.shadow
        update_delta = self.shadow[name] - x
        self.shadow[name] -= (1.0 - self.decay) * update_delta


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, device=torch.device("cuda"), target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.device = device
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real, softlabel):
        if softlabel:
            soft = random.random() * 0.1
        else:
            soft = 0
        if target_is_real:
            target_tensor = self.real_label - soft
        else:
            target_tensor = self.fake_label + soft
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real, softlabel=False):
        target_tensor = self.get_target_tensor(input, target_is_real, softlabel)
        # target_tensor = target_tensor.to(self.device)
        return self.loss(input, target_tensor)


def l2_sim(feature1, feature2):
    Feature = feature1.expand(feature1.size(0), feature1.size(0), feature1.size(1)).transpose(0, 1)
    return torch.norm(Feature - feature2, p=2, dim=2)


class L2ContrastiveLoss(nn.Module):
    """
    Compute L2 contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(L2ContrastiveLoss, self).__init__()
        self.margin = margin

        self.sim = l2_sim

        self.max_violation = max_violation

    def forward(self, feature1, feature2):
        # compute image-sentence score matrix
        scores = self.sim(feature1, feature2)
        # diagonal = scores.diag().view(feature1.size(0), 1)
        diagonal_dist = scores.diag()
        # d1 = diagonal.expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin - scores).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]

        loss = (torch.sum(cost_s ** 2) + torch.sum(diagonal_dist ** 2)) / (2 * feature1.size(0))

        return loss