import os
from os.path import dirname, join, expanduser
import sys
sys.path.append('..')
import torch
import random
import ntpath
import cv2
import glob
from torch.utils.data.sampler import Sampler
from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from torch.utils import data as data_utils
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
from utils import audio
import shutil
import Options_inpainting

hparams = Options_inpainting.Inpainting_Config()

fs = hparams.sample_rate

def create_path(a_path, b_path):
    name_id_path = os.path.join(a_path, b_path)
    if not os.path.exists(name_id_path):
        os.makedirs(name_id_path)
    return name_id_path

def _pad(seq, max_len, constant_values=0):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=constant_values)


def _pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=0)
    return x


def ensure_divisible_mel(length, divisible_by=256, lower=True):
    if length % divisible_by == 0:
        max_steps = length

        return max_steps // audio.get_hop_size()
    if lower:
        max_steps = length - length % divisible_by
    else:
        max_steps = length + (divisible_by - length % divisible_by)

    return max_steps // audio.get_hop_size()


def assert_ready_for_upsampling(x, c):
    assert len(x) % len(c) == 0 and len(x) // len(c) == audio.get_hop_size()


class _NPYDataSource(FileDataSource):
    def __init__(self, data_root, col, speaker_id=None,
                 train=True, test_size=0.05, test_num_samples=None, random_state=1234):
        self.data_root = data_root
        self.col = col
        self.lengths = []
        self.speaker_id = speaker_id
        self.multi_speaker = False
        self.speaker_ids = None
        self.train = train
        self.test_size = test_size
        self.test_num_samples = test_num_samples
        self.random_state = random_state

    def interest_indices(self, paths):
        indices = np.arange(len(paths))
        if self.test_size is None:
            test_size = self.test_num_samples / len(paths)
        else:
            test_size = self.test_size
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=self.random_state)
        return train_indices if self.train else test_indices

    def collect_files(self):
        meta = join(self.data_root, hparams.metadata_name)
        with open(meta, "rb") as f:
            lines = f.readlines()
        l = lines[0].decode("utf-8").split("|")
        assert len(l) == 3 or len(l) == 4
        self.multi_speaker = len(l) == 4
        self.lengths = list(
            map(lambda l: int(l.decode("utf-8").split("|")[2]), lines))

        paths_relative = list(map(lambda l: l.decode("utf-8").split("|")[self.col], lines))
        paths = list(map(lambda f: join(self.data_root, f), paths_relative))

        if self.multi_speaker:
            speaker_ids = list(map(lambda l: int(l.decode("utf-8").split("|")[-1]), lines))
            self.speaker_ids = speaker_ids
            if self.speaker_id is not None:
                # Filter by speaker_id
                # using multi-speaker dataset as a single speaker dataset
                indices = np.array(speaker_ids) == self.speaker_id
                paths = list(np.array(paths)[indices])
                self.lengths = list(np.array(self.lengths)[indices])

                # Filter by train/tset
                indices = self.interest_indices(paths)
                paths = list(np.array(paths)[indices])
                self.lengths = list(np.array(self.lengths)[indices])

                # aha, need to cast numpy.int64 to int
                self.lengths = list(map(int, self.lengths))
                self.multi_speaker = False

                return paths

        # Filter by train/test
        indices = self.interest_indices(paths)
        paths = list(np.array(paths)[indices])
        lengths_np = list(np.array(self.lengths)[indices])
        self.lengths = list(map(int, lengths_np))

        if self.multi_speaker:
            speaker_ids_np = list(np.array(self.speaker_ids)[indices])
            self.speaker_ids = list(map(int, speaker_ids_np))
            assert len(paths) == len(self.speaker_ids)
        paths = sorted(paths)
        with open(os.path.join(hparams.data_root, str(self.train) + '_paths.txt'), 'w') as text_file:
            for l in range(len(paths)):
                text_file.writelines(paths[l] + '\n')
        return paths

    def collect_features(self, path):
        return np.load(path)


class MelSpecDataSource(_NPYDataSource):
    def __init__(self, data_root, **kwargs):
        super(MelSpecDataSource, self).__init__(data_root, 1, **kwargs)


class LoadFlow():
    def __init__(self, path):
        self.path = path
        self.flow_x_path = os.path.join(path, 'flow_x')
        self.flow_x_crop_path = create_path(path, 'flow_x_crop')
        self.flow_y_path = os.path.join(path, 'flow_y')
        self.flow_y_crop_path = create_path(path, 'flow_y_crop')
        self.image_path = os.path.join(path, 'image')
        self.image_crop_path = create_path(path, 'image_crop')

        self.find_all_flows = glob.glob(os.path.join(self.flow_x_path, '*.jpg'))
        self.len_flows = len(self.find_all_flows)


    def __getitem__(self, item):
        flow_x_path = os.path.join(self.flow_x_path, str(item + 1) + '.jpg')
        flow_y_path = os.path.join(self.flow_y_path, str(item + 1) + '.jpg')
        image_path = os.path.join(self.image_path, str(item + 1) + '.jpg')
        flow_x = cv2.imread(flow_x_path, 0)
        flow_y = cv2.imread(flow_y_path, 0)
        image = cv2.imread(image_path)
        return flow_x, flow_y, image

    def __len__(self):
        return len(self.find_all_flows)


def find_cluster(sum_i):
    min_idx = 0
    max_idx = -1
    min_num = sum_i[min_idx]
    max_num = sum_i[max_idx]
    for num in range(len(sum_i) - 1):
        if (min_num + 5) not in sum_i:
            min_idx += 1
            min_num = sum_i[min_idx]
    for num in range(len(sum_i) - 1):
        if (max_num - 5) not in sum_i:
            max_idx -= 1
            max_num = sum_i[max_idx]
    if min_num > max_num:
        max_num = min_num
    return min_num, max_num


def crop_image(load_flow):
    sum_flow_x = 0
    sum_flow_y = 0
    for num in range(load_flow.len_flows):
        flow_x, flow_y, image = load_flow[num]
        sum_flow_x += abs(flow_x.astype(np.int32) - 127)
        sum_flow_y += abs(flow_y.astype(np.int32) - 127)
    Sum_flow = sum_flow_x + sum_flow_y
    if type(Sum_flow) == int:
        w_min_num = w_max_num = h_min_num = h_max_num = 0

    else:
        mask = (Sum_flow > (2 * load_flow.len_flows)).astype(int)
        sum_w = np.where(np.sum(mask, 0) > 0)[0]
        sum_h = np.where(np.sum(mask, 1) > 0)[0]
        if len(sum_w) != 0 and len(sum_h) != 0:

            w_min_num, w_max_num = find_cluster(sum_w)
            h_min_num, h_max_num = find_cluster(sum_h)
        else:
            w_min_num = w_max_num = h_min_num = h_max_num = 0
    return w_min_num, w_max_num, h_min_num, h_max_num


def padding_square(image):
    size = np.shape(image)
    w, h = size[0], size[1]
    if w != h:
        larger_side = max(w, h)
        if larger_side == w:
            delta_h = w - h
            left, right = delta_h // 2, delta_h - (delta_h//2)
            top = bottom = 0
        else:
            delta_h = h - w
            top, bottom = delta_h // 2, delta_h - (delta_h//2)
            left = right = 0
        if len(size) == 3:
            target = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[127, 127, 127])
        else:
            target = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[127])
    else:
        target = image
    return target


class ImageProcess(Dataset):
    def __init__(self, root_path, audio_paths=None, hparams=hparams, mode='train'):
        if audio_paths is not None:
            self.audio_paths = audio_paths
            self.from_txt = True
        else:
            self.audio_paths = sorted(os.listdir(root_path))
            self.from_txt = False
        self.root_path = root_path

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        if self.from_txt:
            file_name = ntpath.basename(audio_path[:-4])
            file_folder_path = os.path.join(self.root_path, file_name)
            if not os.path.isdir(file_folder_path):
                shutil.rmtree(audio_path)
        else:
            file_folder_path = os.path.join(self.root_path, audio_path)

        load_flow = LoadFlow(os.path.join(file_folder_path))
        # if os.path.isfile(os.path.exists(os.path.join(load_flow.image_crop_path, str(len(load_flow)) + '.jpg'))):
        #     pass
        # else:
        w_min_num, w_max_num, h_min_num, h_max_num = crop_image(load_flow)
        if (w_max_num - w_min_num) < 50 or (h_max_num - h_min_num) < 50:
            shutil.rmtree(file_folder_path)
            print("remove {}".format(file_folder_path))
        else:
            for idx in range(load_flow.len_flows):
                flow_x, flow_y, image = load_flow[idx]
                image_crop = image[h_min_num:h_max_num, w_min_num:w_max_num]
                flow_x = flow_x[h_min_num:h_max_num, w_min_num:w_max_num]
                flow_y = flow_y[h_min_num:h_max_num, w_min_num:w_max_num]
                image_crop = padding_square(image_crop)
                flow_x = padding_square(flow_x)
                flow_y = padding_square(flow_y)
                cv2.imwrite(os.path.join(load_flow.image_crop_path, str(idx + 1) + '.jpg'), image_crop)
                cv2.imwrite(os.path.join(load_flow.flow_x_crop_path, str(idx + 1) + '.jpg'), flow_x)
                cv2.imwrite(os.path.join(load_flow.flow_y_crop_path, str(idx + 1) + '.jpg'), flow_y)
        return file_folder_path

    def __len__(self):
        return len(self.audio_paths)


def resave_data():
    data_loaders = {}
    for phase in ["train"]:
        train = phase == "train"
        Mel = MelSpecDataSource(hparams.data_root, speaker_id=hparams.speaker_id,
                                                  train=train,
                                                  test_size=hparams.test_size)

        paths = Mel.collect_files()
        image_loader = ImageProcess(hparams.image_path)
        data_loader = data_utils.DataLoader(
                image_loader, batch_size=1,
                num_workers=hparams.num_workers)
        data_loaders[phase] = data_loader

        for phase, data_loader in data_loaders.items():

            for step, file_folder_path in enumerate(data_loader):
                print('finish processing {}'.format(file_folder_path))


resave_data()
# image = cv2.imread("/home/hzhou/Documents/AV-generation/MUSIC_dataset/data/cello/data/0054_0001/image_crop/5.jpg")
# print(np.shape(image))
