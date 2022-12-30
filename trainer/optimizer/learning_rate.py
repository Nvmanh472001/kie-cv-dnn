from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch.optim import lr_scheduler as lr

class Cosine(object):
    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(Cosine, self).__init__()
        self.learning_rate = learning_rate
        self.T_max = step_each_epoch * epochs
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)

    def __call__(self):
        learning_rate = lr.CosineAnnealingLR(
            learning_rate=self.learning_rate,
            T_max=self.T_max,
            last_epoch=self.last_epoch)
        if self.warmup_epoch > 0:
            learning_rate = lr.LinearLR(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_epoch,
                start_lr=0.0,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
            
        return learning_rate