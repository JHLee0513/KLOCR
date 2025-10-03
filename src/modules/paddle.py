from .base import Module
from paddleocr import PaddleOCR
import time
import cv2
import numpy as np
import torch
from PIL import Image
from .common import group_text_box

def alpha_to_color(img, alpha_color=(255, 255, 255)):
    if len(img.shape) == 3 and img.shape[2] == 4:
        B, G, R, A = cv2.split(img)
        alpha = A / 255

        R = (alpha_color[0] * (1 - alpha) + R * alpha).astype(np.uint8)
        G = (alpha_color[1] * (1 - alpha) + G * alpha).astype(np.uint8)
        B = (alpha_color[2] * (1 - alpha) + B * alpha).astype(np.uint8)

        img = cv2.merge((B, G, R))
    return img


def preprocess_image(_image, alpha_color=(255, 255, 255)):
    _image = alpha_to_color(_image, alpha_color)
    return _image


class PaddleDet(Module):
    def __init__(self, module_args, input_key=None, output_key=None):
        super().__init__(module_args, input_key=input_key, output_key=output_key)
        print(module_args)
        self.model_det = module_args.get('model_det', "model/ch_PP-OCRv4_det_server_infer")
        self.model_reg = module_args.get('model_reg', "model/openatom_rec_svtrv2_ch_infer")
        self.batch_size = module_args.get('batch_size', 30)
        self.drop_score = module_args.get('drop_score', 0.5)
        self.det_db_box_thresh = module_args.get('det_db_box_thresh', 0.5)
        self.gpu_id = module_args.get('gpu_id', 0)
        self.paddle_ocr = PaddleOCR(lang="korean", det_model_dir=self.model_det, cls_model_dir=self.model_reg,
                        use_angle_cls=False, drop_score=self.drop_score,
                        rec_batch_num=self.batch_size, cls_batch_num=self.batch_size,
                        use_space_char=True,
                        gpu_id=self.gpu_id,
                        det_db_box_thresh=self.det_db_box_thresh
                        )
        
        self.verbose = module_args.get('verbose', False)

    def process(self, input_dict):
        if self.verbose:
            start = time.time()
        image = input_dict.get(self.input_key['image'])
        if isinstance(image, Image.Image):
            image = np.asarray(image)
        elif isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        img = preprocess_image(image)
        dt_boxes, elapse = self.paddle_ocr.text_detector(img)
        if dt_boxes.size == 0:
            return {self.output_key['roi']: []}
        if self.verbose:
            print(f"Time Taken [PaddleDet]: {time.time() - start}")
        return {self.output_key['roi']: dt_boxes.astype(int)}

class PaddleRec(Module):
    def __init__(self, module_args, input_key=None, output_key=None):
        global g_paddle_ocr
        super().__init__(module_args, input_key=input_key, output_key=output_key)
        print(module_args)
        self.model_det = module_args.get('model_det', "model/ch_PP-OCRv4_det_server_infer")
        self.model_reg = module_args.get('model_reg', "model/openatom_rec_svtrv2_ch_infer")
        self.batch_size = module_args.get('batch_size', 30)
        self.drop_score = module_args.get('drop_score', 0.5)
        self.device = module_args.get('device', 'cuda')
        assert 'cuda' in self.device, "GPU device must be passed in for Paddle !"
        self.gpu_id = int(self.device.replace('cuda:').strip())
        self.paddle_ocr = PaddleOCR(lang="korean", det_model_dir=self.model_det, cls_model_dir=self.model_reg,
                        use_angle_cls=False, drop_score=self.drop_score,
                        rec_batch_num=self.batch_size, cls_batch_num=self.batch_size,
                        use_space_char=True,
                        gpu_id=self.gpu_id
                        )

    def process(self, input_dict):
        image = input_dict[self.input_key['image']]
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
        roi = input_dict[self.input_key['roi']]

        det_list = []
        polys = []
        for box in roi:
            ys = box[:, 1].astype(int)
            xs = box[:, 0].astype(int)
            xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
            roi_image = image[ymin:ymax, xmin:xmax]
            det_list.append(box)
            # poly top-left, top-right, low-right, low-left
            polys.append((xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax))

        if self.module_args['group_boxes']:
            det_list = group_text_box(polys, slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5, width_ths = 1.0, add_margin = 0.05, sort_output = True)

        return {
            self.output_key['roi']: np.array(det_list)
        }