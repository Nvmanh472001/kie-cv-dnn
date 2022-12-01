__all__ = ['build_backbone']


def build_backbone(config, model_type):
    if model_type == 'det':
        from .detection.mobilenet_v3 import MobileNetV3
    elif model_type == 'rec':
        from .recognition.svtrnet import SVTRNet

    module_name = config.pop('name')
    module_class = eval(module_name)(**config)

    return module_class
