from .bce_loss import BCELoss
from .ctc_loss import CTCLoss
import copy

def build_loss(config):
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    
    module_class = eval(module_name)(**config)
    return module_class