from typing import Dict, Union
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time
from .base import Module

class CheckboxModule(Module):
    """Module for detecting and classifying checkboxes in documents"""
    
    # Define constants
    BOX_COLORS = {
        "unchecked": (242, 48, 48),
        "checked": (38, 115, 101),
        "block": (242, 159, 5)
    }
    BOX_PADDING = 2

    def __init__(self, module_args, input_key=None, output_key=None):
        super().__init__(module_args, input_key=input_key, output_key=output_key)
        self.module_args = module_args
        self.device = self.module_args.get('device', 'cpu')
        self.conf = self.module_args.get('conf', 0.5)
        self.iou = self.module_args.get('iou', 0.8)
        self.verbose = self.module_args.get('verbose', False)
        self.viz = self.module_args.get('viz', False)

        if self.module_args.get('weights'):
            self.model = YOLO(self.module_args.get('weights'))
            self.model.to(self.device)
        else:
            raise ValueError("Weights are required to load checkbox detection model")
        
        self.verbose = self.module_args.get('verbose', False)

    def preprocess(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """Preprocess input image for checkbox detection
        
        Args:
            image: Input image as file path, PIL Image or numpy array
            
        Returns:
            Preprocessed image as numpy array
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        if image is None:
            raise ValueError("Could not load or process input image")
            
        return image

    def process(self, input_dict):
        if self.verbose:
            start = time.time()
        image = input_dict[self.input_key['image']]
        image = self.preprocess(image)
        if self.verbose:
            preprocess_time = time.time() - start
            start = time.time()
        results = self.forward(image)
        if self.verbose:
            forward_time = time.time() - start
            start = time.time()
            print(f"Time Taken [Checkbox]: {preprocess_time, forward_time}")
        return results

    def forward(self, image: np.ndarray) -> Dict:
        """Run checkbox detection inference
        
        Args:
            image: Preprocessed input image
            
        Returns:
            Dictionary containing detection results
        """
        results = self.model.predict(source=image, conf=self.conf, iou=self.iou)
        boxes = results[0].boxes
        
        detections = {
            'roi': [],
            'text': [],
            'conf': []
        }
        for box in boxes:
            # Get detection info
            detection_class_conf = round(box.conf.item(), 2)
            detection_class = list(self.BOX_COLORS)[int(box.cls)]
            
            # Get box coordinates

            start_box = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
            end_box = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))
            
            detections['roi'].append(list(start_box) + list(end_box))
            detections['text'].append(detection_class)
            detections['conf'].append(detection_class_conf)
            
        output_dict = {self.output_key['checkbox_detections']: detections}
        if self.viz:
            output_dict['viz'] = self.postprocess(output_dict, image)
        return output_dict

    def postprocess(self, model_output: Dict, image: np.ndarray) -> np.ndarray:
        """Draw detection results on image
        
        Args:
            model_output: Raw model output dictionary
            image: Original input image
            
        Returns:
            Image with drawn detection boxes and labels
        """
        output_image = image.copy()
        
        for roi, text, conf in zip(self.output_key['checkbox_detections']['roi'], self.output_key['checkbox_detections']['text'], self.output_key['checkbox_detections']['conf']):
            start_box, end_box = roi
            cls = text
            color = self.BOX_COLORS[cls]
            
            # Draw bounding box
            line_thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
            output_image = cv2.rectangle(
                img=output_image,
                pt1=start_box,
                pt2=end_box,
                color=color,
                thickness=line_thickness
            )
            
            # Draw label
            text = f"{cls} {conf}"
            font_thickness = max(line_thickness - 1, 1)
            (text_w, text_h), _ = cv2.getTextSize(
                text=text,
                fontFace=2,
                fontScale=line_thickness/3,
                thickness=font_thickness
            )
            
            # Draw label background
            output_image = cv2.rectangle(
                img=output_image,
                pt1=(start_box[0], start_box[1] - text_h - self.BOX_PADDING*2),
                pt2=(start_box[0] + text_w + self.BOX_PADDING * 2, start_box[1]),
                color=color,
                thickness=-1
            )
            
            # Draw label text
            start_text = (start_box[0] + self.BOX_PADDING, start_box[1] - self.BOX_PADDING)
            output_image = cv2.putText(
                img=output_image,
                text=text,
                org=start_text,
                fontFace=0,
                color=(255,255,255),
                fontScale=line_thickness/3,
                thickness=font_thickness
            )
            
        return output_image
