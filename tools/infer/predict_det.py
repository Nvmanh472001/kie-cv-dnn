import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import numpy as np
import torch
import utils.utilities as utility
from modules.baseocr import BaseOCR
from data.preprocess import create_operators, transform
from data.postprocess import build_postprocess


class TextDetector(BaseOCR):
    def __init__(self, args, **kwargs):
        self.args = args
        self.preprocess_ops = create_operators(args.pre_process_list)
        self.postprocess_ops = build_postprocess(args.postprocess_params)

        use_gpu = args.use_gpu
        self.use_gpu = torch.cuda.is_available() and use_gpu

        self.weights_path = self.weight_path
        self.network_architectures = utility.read_network_config_from_yaml(args.yaml_path)

        super(TextDetector, self).__init__(self.network_architectures, **kwargs)
        self.load_pytorch_weights(self.weights_path)

        self.net.eval()
        if self.use_gpu:
            self.net.cuda()

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_img = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_ops)

        img, shape_list = data
        if img is None:
            return None, 0

        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()

        with torch.no_grad():
            inp = torch.from_numpy(img)
            if self.use_gpu:
                inp = inp.cuda()
            outputs = self.net(inp)

        preds = {}
        preds['maps'] = outputs['maps'].cpu().numpy()

        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_img.shape)

        return dt_boxes
