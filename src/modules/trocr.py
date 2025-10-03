from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from .base import Module
import torch
import numpy as np
import time
import math

class TROCR(Module):

    def __init__(self, module_args, input_key=None, output_key=None):
        """ FeatureExtraction """
        super(TROCR, self).__init__(module_args, input_key=input_key, output_key=output_key)
        model_id = self.module_args['pretrained']
        self.device = self.module_args.get('device', 'cpu')
        self.processor = TrOCRProcessor.from_pretrained(model_id)
        if self.module_args.get('weights'):
            self.model = VisionEncoderDecoderModel.from_pretrained(self.module_args.get('weights')).to(self.device)
        else:
            self.model = VisionEncoderDecoderModel.from_pretrained(model_id).to(self.device)
        self.model_id = model_id
        self.batch_size = self.module_args.get('batch_size', 64)
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.model.config.max_length = self.module_args.get('max_length', 128)
        self.model.config.early_stopping = self.module_args.get('early_stopping', True)
        self.model.config.no_repeat_ngram_size = self.module_args.get('no_repeat_ngram_size', 3)
        self.model.config.length_penalty = self.module_args.get('length_penalty', 2.0)
        self.model.config.num_beams = self.module_args.get('num_beams', 4)
        self.model.config.do_sample = self.module_args.get('do_sample', False)
        
        self.dtype = module_args.get('dtype', 'fp32')
        if self.dtype == 'fp16':
            self.model = self.model.half()
            self.dtype = torch.float16
        elif self.dtype == 'fp32':
            self.dtype = torch.float32
        else:
            raise ValueError(f"Invalid dtype {self.dtype}")
        self.verbose = self.module_args.get('verbose', False)

    def process(self, input_dict):
        if self.verbose:
            start = time.time()
        image = input_dict[self.input_key['image']]
        if isinstance(image, Image.Image):
            image = np.asarray(image)
        elif isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        roi = input_dict[self.input_key['roi']]
        roi = np.asarray(roi)

        crop_list = []

        for box in roi:
            ys = box[:, 1].astype(int)
            xs = box[:, 0].astype(int)
            xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
            roi_image = image[ymin:ymax, xmin:xmax]
            crop_list.append(roi_image)
        if self.verbose:
            box_time = time.time() - start
            start = time.time()
        if len(crop_list) == 0:
            output_dict = {
                self.output_key['text']: []
            }
        
        else: 
            pixel_values = self.processor(crop_list, return_tensors="pt").pixel_values.to(device=self.device, dtype=self.dtype)
            if len(pixel_values) > self.batch_size:
                generated_text = []
                for split_num in range(math.ceil(len(pixel_values) / self.batch_size)):
                    split_start = split_num * self.batch_size
                    split_end = (split_num + 1) * self.batch_size
                    split = pixel_values[split_start:split_end]
                    generated_ids = self.model.generate(split, max_new_tokens=self.model.config.max_length)
                    generated_text_ = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                    generated_text.extend(generated_text_)
            else:
                generated_ids = self.model.generate(pixel_values, max_new_tokens=self.model.config.max_length)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            output_dict = {self.output_key['text']: generated_text}
            if self.verbose:
                model_time = time.time() - start
                start = time.time()
        if self.verbose:
            print(f"Time Taken [TrOCR]: {box_time, model_time}")

        return output_dict