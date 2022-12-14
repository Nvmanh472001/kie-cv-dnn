from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from utils.utilities import generate_label2color
from PIL import Image
import pandas as pd

class LayoutLMv3Cls:
    def __init__(self, args, **kwargs):
        self.args = args
        
        self.config = args["Classification"]
        self.weights_path = self.config["weight_path"]
        self.label_list_path = self.config['label_list_path']
        
        self.labels_list, self.label2id, self.id2label = self.get_labels(self.label_list_path)

        self.processor = LayoutLMv3Processor.from_pretrained(self.weights_path)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(self.weights_path)


    
    def get_labels(self, label_list_path):
        df = pd.read_csv(label_list_path, header=None)
        label_list = df[0].tolist()
        
        id2label = {k: v for k, v in enumerate(label_list)}
        label2id = {v: k for k, v in enumerate(label_list)}
        
        return label_list, label2id, id2label
    
    def encode(self, image, boxes, words):
        encoding = self.processor(images=image,
                                  text=words,
                                  boxes=boxes,
                                  return_tensors='pt',
                                  truncation=True)

        return encoding

    def unnormalize_box(self, bbox, width, height):
        return [
            width * (bbox[0] / 1000),
            height * (bbox[1] / 1000),
            width * (bbox[2] / 1000),
            height * (bbox[3] / 1000),
        ]

    def normalize_box(self, bbox, width, height):
        return [
            int(1000 * bbox[0] / width),
            int(1000 * bbox[1] / height),
            int(1000 * bbox[2] / width),
            int(1000 * bbox[3] / height),
        ]

    def preprocess(self, inputs):
        bboxes = []
        words = []

        ocr_result = inputs['ocr_res']
        width, height = inputs['image_shape']

        for dtboxes, rec_res in ocr_result:
            bbox = dtboxes[1] + dtboxes[3]
            word = rec_res[0]

            bbox = self.normalize_box(bbox, width, height)
            bboxes.append(bbox)
            words.append(word)

        return bboxes, words

    def predict(self, encoding):
        import torch
        with torch.no_grad():
            outputs = self.model(**encoding)

            logits = outputs.logits

            predictions = logits.argmax(-1).squeeze().tolist()
            return predictions

    def draw_image(self, image, bboxes, labels):
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        label2color = generate_label2color(self.labels_list)

        for prediction, box in zip(labels, bboxes):
            draw.rectangle(box, outline=label2color[prediction])
            draw.text((box[0] + 10, box[1] - 10), text=prediction, fill=label2color[prediction], font=font)

        return image

    def __call__(self, image, ocr_result):
        # image = Image.open(image_path)
        ori_img = image.copy()
        width, height = image.size
        preprocess_input = {
            'ocr_res': ocr_result,
            'image_shape': (width, height)
        }

        bboxes, words = self.preprocess(preprocess_input)
        encoding = self.encode(ori_img, bboxes, words)

        predictions = self.predict(encoding)

        true_predictions = [self.model.config.id2label[pred] for pred in predictions]
        true_boxes = [self.unnormalize_box(box, width, height) for box in bboxes]

        img_res = self.draw_image(image, true_boxes, true_predictions)
        return img_res

