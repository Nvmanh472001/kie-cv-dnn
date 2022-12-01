__all__ = ['build_head']


def build_head(config):
    from .detection.db_head import DBHead
    from .recognition.ctc_head import CTCHead

    module_name = config.pop('name')
    module_class = eval(module_name)(**config)

    return module_class
