from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch import optim

class Adam(object):
    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-08,
                 parameter_list=None,
                 weight_decay=None,
                 **kwargs):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.parameter_list = parameter_list
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def __call__(self, model):
        train_params = [
            param for param in model.parameters() if param.trainable is True
        ]

        opt = optim.Adam(
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon,
            weight_decay=self.weight_decay,
            params=train_params
        )
        
        return opt
