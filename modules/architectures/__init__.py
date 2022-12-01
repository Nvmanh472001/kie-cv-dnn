import copy

__all__ = ['build_model']


def build_model(config, **kwargs):
    from .base_model import BaseModel

    config = copy.deepcopy(config)
    module_class = BaseModel(config, **kwargs)

    return module_class
