from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import yaml
import torch

from data import build_dataloader, build_postprocess
from modules.architectures import build_model

from .losses import build_loss
from .optimizer import build_optimizer

class Trainer(object):
    def __init__(self, config) -> None:
        global_config = config['Global']
        
        train_dataloader = build_dataloader(config, 'Train')
        eval_dataloader = build_dataloader(config, 'Eval')
        
        post_process_class = build_postprocess(config['PostProcess'],
                                            global_config)
        
        model = build_model(config['Architecture'])
        
        loss_class = build_loss(config['Loss'])
        
        optimizer, lr_scheduler = build_optimizer(
            config['Optimizer'],
            epochs=config['Global']['epoch_num'],
            step_each_epoch=len(train_dataloader),
            model=model
        )
        
        eval_class = build_metric(config['Metric'])
