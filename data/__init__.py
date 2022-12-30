from .preprocess import transform, create_operators
from .postprocess import build_postprocess

from .dataset import OCRDataSet

import copy
from torch.utils.data import Dataset, DataLoader, BatchSampler

def build_dataloader(configs, mode):
    config = copy.deepcopy(config)
    
    dataset = OCRDataSet(configs, mode)
    
    loader_config = config[mode]['loader']
    batch_size = loader_config['batch_size_per_card']
    drop_last = loader_config['drop_last']
    shuffle = loader_config['shuffle']
    
    num_workers = loader_config['num_workers']
    
    batch_sampler = BatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
    )
    
    if 'collate_fn' in loader_config:
        from .dataset import collate_fn
        collate_fn = getattr(collate_fn, loader_config['collate_fn'])()
    else:
        collate_fn = None
        
    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    
    return data_loader