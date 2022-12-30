from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import copy

from . import regularizer, optimizer, learning_rate

def build_optimizer(config, epochs, step_each_epoch, model):
    config = copy.deepcopy(config)
    
    lr_config = config.pop('lr').update({'epochs': epochs, 'step_each_epoch': step_each_epoch})
    lr = learning_rate.Cosine(**lr_config)
    
    regularizer_config = config.pop('regularizer')
    regularizer_name = regularizer_config.pop('name')
    
    reg = getattr(regularizer, regularizer_name)(**regularizer_config)()
    
    optim_name = config.pop('name')
    
    optim = getattr(optimizer, optim_name)(learning_rate=lr,
                                           weight_decay=reg,
                                           **config)
    
    return optim(model), lr