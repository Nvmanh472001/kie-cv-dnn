import copy

__all__ = ['build_postprocess']


def build_postprocess(config):
    from .db_postprocess import DBPostProcess
    from .ctc_postprocess import CTCLabelDecode

    config = copy.deepcopy(config)

    module_name = config.pop('name')
    module_class = eval(module_name)(**config)

    return module_class
