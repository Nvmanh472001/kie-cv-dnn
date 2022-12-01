__all__ = ['build_transform']


def build_transform(config):
    from .recognition.stn import STN_ON

    module_name = config.pop('name')
    module_class = eval(module_name)(**config)

    return module_class
