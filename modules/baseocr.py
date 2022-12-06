import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import torch

from modules.architectures.base_model import BaseModel


class BaseOCR:
    def __init__(self, config, **kwargs):
        self.config = config
        self.build_net(**kwargs)
        self.net.eval()

    def build_net(self, **kwargs):
        self.net = BaseModel(self.config, **kwargs)

    def load_pytorch_weights(self, weights_path):
        self.net.load_state_dict(torch.load(weights_path))
        print('model is loaded: {}'.format(weights_path))

    def read_pytorch_weights(self, weights_path):
        if not os.path.exists(weights_path):
            raise FileNotFoundError('{} is not existed.'.format(weights_path))
        weights = torch.load(weights_path)
        return weights

    def load_state_dict(self, weights):
        self.net.load_state_dict(weights)
        print('weighs is loaded.')

    def save_pytorch_weights(self, weights_path):
        try:
            torch.save(self.net.state_dict(), weights_path, _use_new_zipfile_serialization=False)
        except:
            torch.save(self.net.state_dict(), weights_path)  # _use_new_zipfile_serialization=False for torch>=1.6.0
        print('model is saved: {}'.format(weights_path))

    def print_pytorch_state_dict(self):
        print('pytorch:')
        for k, v in self.net.state_dict().items():
            print('{}----{}'.format(k, type(v)))

    def get_out_channels(self, weights):
        if list(weights.keys())[-1].endswith('.weight') and len(list(weights.values())[-1].shape) == 2:
            out_channels = list(weights.values())[-1].numpy().shape[1]
        else:
            out_channels = list(weights.values())[-1].numpy().shape[0]
        return out_channels

    def inference(self, inputs):
        with torch.no_grad():
            infer = self.net(inputs)
        return infer
