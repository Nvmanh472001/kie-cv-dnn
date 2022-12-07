import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

from PIL import Image
import cv2
import numpy as np
import torch
import math
from modules.baseocr import BaseOCR
import utils.utilities as utility
from data.postprocess import build_postprocess

class TextRecognizer(BaseOCR):
    def __init__(self, args, **kwargs):
        self.args = args

        rec_configs = args['Recognition']

        global_config = args['Global']
        yaml_path = rec_configs['yaml_path']

        self.rec_image_shape = [int(v) for v in rec_configs['image_shape'].split(",")]
        self.character_type = rec_configs['char_type']
        self.rec_batch_num = rec_configs['batch_num']
        self.max_text_length = rec_configs['max_text_length']

        postprocess_params = utility.get_config_from_yaml(yaml_path['Postprocess'])
        self.postprocess_op = build_postprocess(postprocess_params)

        use_gpu = global_config['use_gpu']
        self.use_gpu = torch.cuda.is_available() and use_gpu

        self.limited_max_width = rec_configs['limited_max_width']
        self.limited_min_width = rec_configs['limited_min_width']

        self.weights_path = rec_configs['weight_path']
        weights = self.read_pytorch_weights(self.weights_path)

        self.network_architectures = utility.read_network_config_from_yaml(yaml_path['Architecture'])

        self.out_channels = self.get_out_channels(weights)

        kwargs['out_channels'] = self.out_channels
        super(TextRecognizer, self).__init__(self.network_architectures, **kwargs)

        self.load_state_dict(weights)
        self.net.eval()

        if self.use_gpu:
            self.net.cuda()

    def resize_norm_img(self, img, image_shape):
        imgC, imgH, imgW = image_shape
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        return resized_image

    def __call__(self, img_list):
        img_num = len(img_list)

        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))

        indices = np.argsort(np.array(width_list))

        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num

        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0

            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)

            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                     self.rec_image_shape)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)

            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            with torch.no_grad():
                inp = torch.from_numpy(norm_img_batch)
                if self.use_gpu:
                    inp = inp.cuda()
                prob_out = self.net(inp)

            if isinstance(prob_out, list):
                preds = [v.cpu().numpy() for v in prob_out]

            else:
                preds = prob_out.cpu().numpy()

            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]

        return rec_res
