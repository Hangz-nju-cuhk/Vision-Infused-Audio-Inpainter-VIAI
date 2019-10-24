import os
from os.path import dirname, join, expanduser
import sys
sys.path.append('..')
import torch
import random
from torch.utils.data.sampler import Sampler
from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from torch.utils import data as data_utils
import glob
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import cv2
from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw, is_scalar_input
import numpy as np
from utils import audio
import Options_inpainting

hparams = Options_inpainting.Inpainting_Config()

fs = hparams.sample_rate


def _pad(seq, max_len, constant_values=0):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=constant_values)


def _pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=0)
    return x


def ensure_divisible(length, divisible_by=256, lower=True):
    if length % divisible_by == 0:
        return length
    if lower:
        return length - length % divisible_by
    else:
        return length + (divisible_by - length % divisible_by)


def assert_ready_for_upsampling(x, c):
    assert len(x) % len(c) == 0 and len(x) // len(c) == audio.get_hop_size()


class _NPYDataSource(FileDataSource):
    def __init__(self, data_root, col, speaker_id=None,
                 train=True, test_size=0.05, test_num_samples=None, random_state=1234):
        self.data_root = data_root
        self.col = col
        self.lengths = []
        self.speaker_id = speaker_id
        self.multi_speaker = True
        self.speaker_ids = None
        self.train = train
        self.test_size = test_size
        self.test_num_samples = test_num_samples
        self.random_state = random_state


    def collect_files(self):
        if self.train:
            metadata = "train" + hparams.new_split_name
        else:
            metadata = "test" + hparams.new_split_name
        meta = join(self.data_root, metadata)
        with open(meta, "rb") as f:
            lines = f.readlines()
        l = lines[0].decode("utf-8").split("|")
        assert len(l) == 4 or len(l) == 5
        self.lengths = list(
            map(lambda l: int(l.decode("utf-8").split("|")[-1]) * 1280, lines))

        paths_relative = list(map(lambda l: l.decode("utf-8").split("|")[self.col], lines))
        paths = list(map(lambda f: join(self.data_root, f), paths_relative))

        if self.multi_speaker:
            speaker_ids = list(map(lambda l: int(l.decode("utf-8").split("|")[-2]), lines))
            self.speaker_ids = speaker_ids
            if self.speaker_id is not None:
                # Filter by speaker_id
                # using multi-speaker dataset as a single speaker dataset
                indices = np.array(speaker_ids) == self.speaker_id
                paths = list(np.array(paths)[indices])

                # Filter by train/tset
                self.lengths = list(np.array(self.lengths)[indices])

                # aha, need to cast numpy.int64 to int
                self.lengths = list(map(int, self.lengths))
                self.multi_speaker = False

                return paths

        # Filter by train/test
        paths = list(np.array(paths))
        lengths_np = list(np.array(self.lengths))
        self.lengths = list(map(int, lengths_np))

        if self.multi_speaker:
            speaker_ids_np = list(np.array(self.speaker_ids))
            self.speaker_ids = list(map(int, speaker_ids_np))
            assert len(paths) == len(self.speaker_ids)
        paths = sorted(paths)
        return paths

    def collect_features(self, path):
        return np.load(path)


class ImageDataSource(FileDataSource):
    def __init__(self, data_root, col, speaker_id=None,
                 train=True, test_size=0.05, test_num_samples=None, random_state=1234):
        self.data_root = data_root
        self.col = col
        self.lengths = []
        self.speaker_id = speaker_id
        self.multi_speaker = True
        self.speaker_ids = None
        self.train = train
        self.test_size = test_size
        self.test_num_samples = test_num_samples
        self.random_state = random_state

    def collect_files(self):
        if self.train:
            metadata = "train" + hparams.new_split_name
        else:
            metadata = "test" + hparams.new_split_name
        meta = join(self.data_root, metadata)
        with open(meta, "rb") as f:
            lines = f.readlines()
        l = lines[0].decode("utf-8").split("|")
        assert len(l) == 4 or len(l) == 5
        self.lengths = list(
            map(lambda l: int(l.decode("utf-8").split("|")[-1]), lines))

        paths_relative = list(map(lambda l: l.decode("utf-8").split("|")[self.col], lines))
        paths = list(map(lambda f: join(self.data_root, f), paths_relative))

        # Filter by train/test
        paths = list(np.array(paths))
        lengths_np = list(np.array(self.lengths))
        self.lengths = list(map(int, lengths_np))

        paths = sorted(paths)
        return paths

    def collect_features(self, path):
        video_block, flow_block, start = sample_data_new(path, self.train, hparams=hparams)
        return video_block, flow_block, start, path


def sample_data_new(data_path, train=True, hparams=hparams):
    flow_x_crop_path = os.path.join(data_path, 'flow_x_crop')
    flow_y_crop_path = os.path.join(data_path, 'flow_y_crop')
    image_crop_path = os.path.join(data_path, 'image_crop')
    find_all_flows = glob.glob(os.path.join(flow_x_crop_path, '*.jpg'))
    len_flows = len(find_all_flows)
    max_time_steps = hparams.max_time_steps
    num_images = len_flows
    max_time_second = max_time_steps / hparams.sample_rate

    use_image_num = int(np.floor(max_time_second / (0.04 * hparams.image_hope_size)))
    image_start = np.random.randint(25, num_images - use_image_num - 25 + 1)

    # assert hparams.load_num > 0
    start = []
    start.append(image_start)
    for ln in range(1, hparams.load_num):
        random1 = np.random.randint(0, image_start - 25 + 1)
        random2 = np.random.randint(image_start + 25, num_images - use_image_num + 1)
        if np.random.randint(0, 2) == 1:
            if random1 - start[-1] > 10:
                start.append(random1)
            else:
                start.append(random2)
        else:
            if random2 - start[-1] > 10:
                start.append(random2)
            else:
                start.append(random1)
    image_rescal_size = hparams.image_rescal_size
    image_size = hparams.image_size
    if train:
        video_block = np.zeros(
                (hparams.load_num, use_image_num, hparams.image_rescal_size, hparams.image_rescal_size, 3))
        flow_block = np.zeros(
                (hparams.load_num, use_image_num, hparams.image_rescal_size, hparams.image_rescal_size, 2))
        crop_x = np.random.randint(0, image_rescal_size - image_size)
        crop_y = np.random.randint(0, image_rescal_size - image_size)
        flip = np.random.randint(0, 2)
    else:
        video_block = np.zeros(
                (hparams.load_num, use_image_num, hparams.image_size, hparams.image_size, 3))
        flow_block = np.zeros(
                (hparams.load_num, use_image_num, hparams.image_size, hparams.image_size, 2))
        crop_x = 0
        crop_y = 0
    if hparams.image or hparams.flow:
        for ln in range(hparams.load_num):
            i = 0
            for item in range(start[ln], use_image_num + start[ln]):
                flow_x_path = os.path.join(flow_x_crop_path, str(item + 1) + '.jpg')
                flow_y_path = os.path.join(flow_y_crop_path, str(item + 1) + '.jpg')
                image_path = os.path.join(image_crop_path, str(item + 1) + '.jpg')
                if hparams.image:
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if train:
                        image = cv2.resize(image, (image_rescal_size, image_rescal_size))
                        if flip:
                            image = np.fliplr(image)
                    else:
                        image = cv2.resize(image, (image_size, image_size))
                    image = (image - 127.) / 128.
                    video_block[ln, i, :] = image
                if hparams.flow:
                    flow_x = cv2.imread(flow_x_path, 0)
                    flow_y = cv2.imread(flow_y_path, 0)
                    if train:
                        flow_x = cv2.resize(flow_x, (image_rescal_size, image_rescal_size))
                        flow_y = cv2.resize(flow_y, (image_rescal_size, image_rescal_size))

                        if flip:
                            flow_x = np.fliplr(flow_x)
                            flow_y = np.fliplr(flow_y)
                    else:
                        flow_x = cv2.resize(flow_x, (image_size, image_size))
                        flow_y = cv2.resize(flow_y, (image_size, image_size))
                    flow_y = (flow_y - 127.) / 128.
                    flow_x = (flow_x - 127.) / 128.

                    flow_block[ln, i, :, :, 0] = flow_x
                    flow_block[ln, i, :, :, 1] = flow_y
                i += 1
    video_block = video_block[:, :, crop_x:crop_x + image_size,
                  crop_y:crop_y + image_size]
    flow_block = flow_block[:, :, crop_x:crop_x + image_size,
                 crop_y:crop_y + image_size]
    video_block = video_block.transpose((0, 1, 4, 2, 3))
    flow_block = flow_block.transpose((0, 1, 4, 2, 3))
    return video_block, flow_block, start


def load_image(path, train, hparams=hparams):
    flow_x_crop_path = os.path.join(path, 'flow_x_crop')
    flow_y_crop_path = os.path.join(path, 'flow_y_crop')
    image_crop_path = os.path.join(path, 'image_crop')
    find_all_flows = glob.glob(os.path.join(flow_x_crop_path, '*.jpg'))
    len_flows = len(find_all_flows)
    image_rescal_size = hparams.image_rescal_size
    image_size = hparams.image_size
    if train:
        video_block = np.zeros(
                (len_flows, hparams.image_rescal_size, hparams.image_rescal_size, 3))
        flow_block = np.zeros(
                (len_flows,  hparams.image_rescal_size, hparams.image_rescal_size, 2))
        crop_x = np.random.randint(0, image_rescal_size - image_size)
        crop_y = np.random.randint(0, image_rescal_size - image_size)
        flip = np.random.randint(0, 2)
    else:
        video_block = np.zeros(
                (hparams.load_num, hparams.image_size, hparams.image_size, 3))
        flow_block = np.zeros(
                (hparams.load_num, hparams.image_size, hparams.image_size, 2))
        crop_x = 0
        crop_y = 0
    i = 0
    for item in range(len_flows):
        flow_x_path = os.path.join(flow_x_crop_path, str(item + 1) + '.jpg')
        flow_y_path = os.path.join(flow_y_crop_path, str(item + 1) + '.jpg')
        image_path = os.path.join(image_crop_path, str(item + 1) + '.jpg')
        if hparams.image:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if train:
                image = cv2.resize(image, (image_rescal_size, image_rescal_size))
                if flip:
                    image = np.fliplr(image)
            else:
                image = cv2.resize(image, (image_size, image_size))
            image = (image - 127.) / 128.
            video_block[i] = image
        if hparams.flow:
            flow_x = cv2.imread(flow_x_path, 0)
            flow_y = cv2.imread(flow_y_path, 0)
            if train:
                flow_x = cv2.resize(flow_x, (image_rescal_size, image_rescal_size))
                flow_y = cv2.resize(flow_y, (image_rescal_size, image_rescal_size))

                if flip:
                    flow_x = np.fliplr(flow_x)
                    flow_y = np.fliplr(flow_y)
            else:
                flow_x = cv2.resize(flow_x, (image_size, image_size))
                flow_y = cv2.resize(flow_y, (image_size, image_size))
            flow_y = (flow_y - 127.) / 128.
            flow_x = (flow_x - 127.) / 128.

            flow_block[i, :, :, 0] = flow_x
            flow_block[i, :, :, 1] = flow_y
        i += 1
    video_block = video_block[:, crop_x:crop_x + image_size,
                  crop_y:crop_y + image_size,:]
    flow_block = flow_block[:, crop_x:crop_x + image_size,
                 crop_y:crop_y + image_size,:]
    video_block = video_block.transpose((0, 3, 1, 2))
    flow_block = flow_block.transpose((0, 3, 1, 2))
    return video_block, flow_block


class RawAudioDataSource(_NPYDataSource):
    def __init__(self, data_root, **kwargs):
        super(RawAudioDataSource, self).__init__(data_root, 2, **kwargs)


class MelSpecDataSource(_NPYDataSource):
    def __init__(self, data_root, **kwargs):
        super(MelSpecDataSource, self).__init__(data_root, 1, **kwargs)


class ImageSpecDataSource(ImageDataSource):
    def __init__(self, data_root, **kwargs):
        super(ImageSpecDataSource, self).__init__(data_root, 0, **kwargs)


class PartialyRandomizedSimilarTimeLengthSampler(Sampler):
    """Partially randomized sampler

    1. Sort by lengths
    2. Pick a small patch and randomize it
    3. Permutate mini-batches
    """

    def __init__(self, lengths, batch_size=16, batch_group_size=None,
                 permutate=True):
        self.lengths, self.sorted_indices = torch.sort(torch.LongTensor(lengths))

        self.batch_size = batch_size
        if batch_group_size is None:
            batch_group_size = min(batch_size * 32, len(self.lengths))
            if batch_group_size % batch_size != 0:
                batch_group_size -= batch_group_size % batch_size

        self.batch_group_size = batch_group_size
        assert batch_group_size % batch_size == 0
        self.permutate = permutate

    def __iter__(self):
        indices = self.sorted_indices.clone()
        batch_group_size = self.batch_group_size
        s, e = 0, 0
        for i in range(len(indices) // batch_group_size):
            s = i * batch_group_size
            e = s + batch_group_size
            random.shuffle(indices[s:e])

        # Permutate batches
        if self.permutate:
            perm = np.arange(len(indices[:e]) // self.batch_size)
            random.shuffle(perm)
            indices[:e] = indices[:e].view(-1, self.batch_size)[perm, :].view(-1)

        # Handle last elements
        s += batch_group_size
        if s < len(indices):
            random.shuffle(indices[s:])

        return iter(indices)

    def __len__(self):
        return len(self.sorted_indices)


class PyTorchDataset(object):
    def __init__(self, X, Mel):
        self.X = X
        self.Mel = Mel
        # alias
        self.multi_speaker = X.file_data_source.multi_speaker

    def __getitem__(self, idx):
        if self.Mel is None:
            mel = None
        else:
            mel = self.Mel[idx]

        raw_audio = self.X[idx]
        if self.multi_speaker:
            speaker_id = self.X.file_data_source.speaker_ids[idx]
        else:
            speaker_id = None

        # (x,c,g)
        return raw_audio, mel, speaker_id

    def __len__(self):
        return len(self.X)


class PyTorchImageDataset(object):
    def __init__(self, X, Mel, Image):
        self.X = X
        self.Mel = Mel
        self.Image = Image
        # alias
        self.multi_speaker = X.file_data_source.multi_speaker

    def __getitem__(self, idx):
        if self.Mel is None:
            mel = None
        else:
            mel = self.Mel[idx]

        raw_audio = self.X[idx]
        video_block, flow_block, start, path = self.Image[idx]
        if self.multi_speaker:
            speaker_id = self.X.file_data_source.speaker_ids[idx]
        else:
            speaker_id = None

        # (x,c,g)
        return raw_audio, mel, video_block, flow_block, start, speaker_id, path

    def __len__(self):
        return len(self.X)


def collate_fn(batch):
    """Create batch

    Args:
        batch(tuple): List of tuples
            - x[0] (ndarray,int) : list of (T,)
            - x[1] (ndarray,int) : list of (T, D)
            - x[2] (ndarray,int) : list of (1,), speaker id
    Returns:
        tuple: Tuple of batch
            - x (FloatTensor) : Network inputs (B, C, T)
            - y (LongTensor)  : Network targets (B, T, 1)
    """

    local_conditioning = len(batch[0]) >= 2 and hparams.cin_channels > 0
    global_conditioning = len(batch[0]) >= 3 and hparams.file_channel > 0

    if hparams.max_time_sec is not None:
        max_time_steps = int(hparams.max_time_sec * hparams.sample_rate)
    elif hparams.max_time_steps is not None:
        max_time_steps = hparams.max_time_steps
    else:
        max_time_steps = None
    max_time_second = max_time_steps / hparams.sample_rate

    use_image_num = int(np.floor(max_time_second / (0.04 * hparams.image_hope_size)))
    # Time resolution adjustment
    video_block = []
    flow_block = []
    if local_conditioning:
        new_batch = []
        for idx in range(len(batch)):
            x, c, video, flow, start, g, path = batch[idx]
            if hparams.upsample_conditional_features:
                assert_ready_for_upsampling(x, c)
                if max_time_steps is not None:
                    max_steps = ensure_divisible(max_time_steps, audio.get_hop_size(), True)
                    if len(x) > max_steps:

                        for ln in range(hparams.load_num):
                            mel_start = 3 + 4 * start[ln]
                            c1 = c[mel_start:mel_start + use_image_num * 4]
                            x1 = x[mel_start * hparams.hop_size : (mel_start + use_image_num * 4) * hparams.hop_size]
                            new_batch.append((x1, c1, g, os.path.join(path, str(start[ln]))))
                        video_block.append(torch.FloatTensor(video))
                        flow_block.append(torch.FloatTensor(flow))
        batch = new_batch

    # Lengths
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = max(input_lengths)

    # (B, T, C)
    # pad for time-axis
    if is_mulaw_quantize(hparams.input_type):
        x_batch = np.array([_pad_2d(np_utils.to_categorical(
            x[0], num_classes=hparams.quantize_channels),
            max_input_len) for x in batch], dtype=np.float32)
    else:
        x_batch = np.array([_pad_2d(x[0].reshape(-1, 1), max_input_len)
                            for x in batch], dtype=np.float32)
    assert len(x_batch.shape) == 3

    # (B, T)
    if is_mulaw_quantize(hparams.input_type):
        y_batch = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int)
    else:
        y_batch = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.float32)
    assert len(y_batch.shape) == 2

    # (B, T, D)
    if local_conditioning:
        max_len = max([len(x[1]) for x in batch])
        c_batch = np.array([_pad_2d(x[1], max_len) for x in batch], dtype=np.float32)
        assert len(c_batch.shape) == 3
        # (B x C x T)
        c_batch = torch.FloatTensor(c_batch).transpose(1, 2).contiguous()
    else:
        c_batch = None

    if global_conditioning:
        g_batch = torch.LongTensor([x[2] for x in batch])
    else:
        g_batch = None

    path_batch = list(x[3] for x in batch)

    video_batch = torch.cat(video_block, 0)
    flow_batch = torch.cat(flow_block, 0)

    # Covnert to channel first i.e., (B, C, T)
    x_batch = torch.FloatTensor(x_batch).transpose(1, 2).contiguous()
    # Add extra axis
    if is_mulaw_quantize(hparams.input_type):
        y_batch = torch.LongTensor(y_batch).unsqueeze(-1).contiguous()
    else:
        y_batch = torch.FloatTensor(y_batch).unsqueeze(-1).contiguous()

    input_lengths = torch.LongTensor(input_lengths)

    return video_batch, flow_batch, c_batch, x_batch, y_batch, g_batch, input_lengths, path_batch



def get_data_loaders(data_root, speaker_id=None, test_shuffle=True):
    data_loaders = {}
    local_conditioning = hparams.cin_channels > 0
    for phase in ["train", "test"]:
        train = phase == "train"
        X = FileSourceDataset(RawAudioDataSource(data_root, speaker_id=speaker_id,
                                                 train=train,
                                                 test_size=hparams.test_size))
        Image = FileSourceDataset(ImageSpecDataSource(data_root, speaker_id=speaker_id,
                                                      train=train,
                                                      test_size=hparams.test_size))
        if local_conditioning:
            Mel = FileSourceDataset(MelSpecDataSource(data_root, speaker_id=speaker_id,
                                                      train=train,
                                                      test_size=hparams.test_size))
            assert len(X) == len(Mel)
            print("Local conditioning enabled. Shape of a sample: {}.".format(
                Mel[0].shape))
        else:
            Mel = None
        print("[{}]: length of the dataset is {}".format(phase, len(X)))

        if train:
            lengths = np.array(X.file_data_source.lengths)
            # Prepare sampler
            sampler = PartialyRandomizedSimilarTimeLengthSampler(
                lengths, batch_size=hparams.batch_size)
            shuffle = True
        else:
            sampler = None
            shuffle = test_shuffle

        dataset = PyTorchImageDataset(X, Mel, Image)

        data_loader = data_utils.DataLoader(
            dataset, batch_size=hparams.batch_size,
            num_workers=hparams.num_workers, shuffle=shuffle,
            collate_fn=collate_fn, pin_memory=hparams.pin_memory)

        speaker_ids = {}

        if len(speaker_ids) > 0:
            print("Speaker stats:", speaker_ids)

        data_loaders[phase] = data_loader

    return data_loaders

