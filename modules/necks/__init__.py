__all__ = ['build_neck']


def build_neck(config):
    from .detection.rse_fpn import RSEFPN
    from .recognition.rnn import SequenceEncoder

    module_name = config.pop('name')
    module_class = eval(module_name)(**config)

    return module_class
