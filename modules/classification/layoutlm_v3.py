from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification


class LayoutLMv3:
    def __init__(self, weights_path, config, **kwargs):
        self.weights_path = weights_path
        self.config = config

        self.processor = LayoutLMv3Processor.from_pretrained(weights_path)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(weights_path)

    def _encode(self, image, words, boxes, word_labels):
        encoding = self.processor(image, words,
                                  boxes=boxes, word_labels=word_labels,
                                  return_tensors='pt', truncation=True)

        encode_value = {
            'value': encoding,
            'image_shape': image.size
        }

        return encode_value

    def _unnormalize_box(self, bbox, width, height):
        return [
            width * (bbox[0] / 1000),
            height * (bbox[1] / 1000),
            width * (bbox[2] / 1000),
            height * (bbox[3] / 1000),
        ]

    def _predict(self, encode_value):
        import torch
        with torch.no_grad():
            encoding = encode_value['value']
            width, height = encoding['image_shape']

            outputs = self.model(**encoding)

            logits = outputs.logits

            predictions = logits.argmax(-1).squeeze().tolist()
            labels = encoding.labels.squeeze().tolist()

            token_boxes = encoding.bbox.squeeze().tolist()

            true_predictions = [self.model.config.id2label[pred] for pred, label in zip(predictions, labels)
                                if label != - 100]

            true_boxes = [self._unnormalize_box(box, width, height) for box, label in zip(token_boxes, labels)
                          if label != -100]

            return ( true_predictions, true_boxes )

    def _draw_image(self, image):
        pass