import argparse
import os

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default='VIAI-AV', help='The name of the model')
        self.parser.add_argument('--data_root', type=str, help='The main folder path of the class')
        self.parser.add_argument('--resume', action='store_true', help='Whether to load a checkpoint')
        self.parser.add_argument('--resume_path', type=str, default=None, help='The checkpoint path for the main framework')
        self.parser.add_argument('--wavenet_pretrain', type=str, default=None, help='The wavenet pretrain path')
        self.parser.add_argument('--new_split_name', type=str, default="_new_split.txt", help='The name of the txt with the train/test split information')

        self.parser.add_argument('--num_workers', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--feature_length', type=int, default=256, help='feature length')
        self.parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        self.parser.add_argument('--image_size', type=int, default=256, help='scale images to this size')
        self.parser.add_argument('--image_channel_size', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--mfcc_width', type=int, default=12, help='width of loaded mfcc feature')
        self.parser.add_argument('--mfcc_length', type=int, default=20, help='length of loaded mfcc feature')
        self.parser.add_argument('--image_block_name', type=str, default='align_face256',
                                 help='training folder name containing images')
        self.parser.add_argument('--disfc_length', type=int, default=20,
                                 help='# of frames sending into the discriminate fc')
        self.parser.add_argument('--mul_gpu', type=bool, default=True, help='whether to use mul gpus')

        self.parser.add_argument('--cuda_on', type=bool, default=True, help='whether to use gpu')