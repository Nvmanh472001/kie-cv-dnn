from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class BCELoss(nn.Layer):
    def __init__(self, reduction='mean'):
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, label, mask=None, weight=None, name=None):
        loss = F.binary_cross_entropy(input, label, reduction=self.reduction)
        return loss
