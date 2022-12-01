from transformers import LayoutLMv3Model, LayoutLMv3Processor


class LayoutLMv3:
    def __init__(self, weights_path, config, **kwargs):
        self.weights_path = weights_path
        self.config = config

        self.processor = LayoutLMv3Processor.from_pretrained(weights_path)
