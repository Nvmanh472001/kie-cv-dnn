import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import cv2
import copy
import numpy as np
from PIL import Image
from tools.infer.predict_det import TextDetector
from tools.infer.predict_rec import TextRecognizer
from modules.classification import LayoutLMv3Cls
import utils.utilities as utility

class TextSystem(object):
    def __init__(self, args, **kwargs):
        self.text_detector = TextDetector(args, **kwargs)
        self.text_recognizer = TextRecognizer(args, **kwargs)
        self.text_classification = LayoutLMv3Cls(args, **kwargs)

    def get_rotate_crop_image(self, img, points):
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])
            )
        )

        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])
            )
        )

        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])

        M = cv2.getPerspectiveTransform(points, pts_std)

        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC
        )

        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def __call__(self, img):
        ori_img = img.copy()
        img_pil = Image.fromarray(img)
        
        dt_boxes = self.text_detector(ori_img)
        print("dt_boxes num : {}".format(len(dt_boxes)))

        if dt_boxes is None:
            return None, None

        img_crop_list = []
        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_img, tmp_box)
            img_crop_list.append(img_crop)

        dtboxes = (list(map(lambda np_ndarray: np_ndarray.tolist(), dt_boxes)))
        rec_res = self.text_recognizer(img_crop_list)
        
        ocr_result = zip(dtboxes, rec_res)
        img_res = self.text_classification(image=img_pil, ocr_result=ocr_result)
        return img_res
        


def sorted_boxes(dt_boxes):
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
            
    return _boxes