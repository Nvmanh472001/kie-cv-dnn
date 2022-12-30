from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch


class L1Decay(object):
    def __init__(self, factor=0.0):
        super(L1Decay, self).__init__()
        self.coeff = factor

    def __call__(self):
        reg = torch.regularizer.L1Decay(self.coeff)
        return reg


class L2Decay(object):
    def __init__(self, factor=0.0):
        super(L2Decay, self).__init__()
        self.coeff = float(factor)

    def __call__(self):
        return self.coeff