# PixParse Dataset processing
# example: python process_pixparse.py --source path/to/idl-wds --destination path/to/idl-wds-processed


import json
import cv2
import argparse
import os
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
import traceback

# if desired, randomly select portion of data since the original dataset is huge
SAMPLE_RATE = 1.0

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--destination", type=str, required=True)

    args = parser.parse_args()
    src = args.source
    dest = args.destination
    files = np.unique([x.split(".")[0] for x in os.listdir(src) if os.path.isfile(os.path.join(src, x)) and x.split(".")[1] == "json" and os.path.isfile(os.path.join(src, x.split(".")[0] + ".pdf"))])

    if SAMPLE_RATE < 1.0:
        subsample = np.random.choice(files, size=int(len(files) * SAMPLE_RATE), replace=False)
    else:
        subsample = files
    # split it into 80/20 for training/validation
    train_list, val_list = train_test_split(subsample, test_size=0.2, random_state=2401)
    print(f"Processing {len(train_list)} files for Train and {len(val_list)} files for Validation..")

    tr_save_split_path = os.path.join(dest, 'Training')
    tr_save_split_img = os.path.join(tr_save_split_path, "images")
    tr_save_split_lbl = os.path.join(tr_save_split_path, "label")
    os.makedirs(tr_save_split_path, exist_ok=True)
    os.makedirs(tr_save_split_img, exist_ok=True)
    os.makedirs(tr_save_split_lbl, exist_ok=True)

    vl_save_split_path = os.path.join(dest, 'Validation')
    vl_save_split_img = os.path.join(vl_save_split_path, "images")
    vl_save_split_lbl = os.path.join(vl_save_split_path, "label")
    os.makedirs(vl_save_split_path, exist_ok=True)
    os.makedirs(vl_save_split_img, exist_ok=True)
    os.makedirs(vl_save_split_lbl, exist_ok=True)

    for fn in tqdm(train_list):
        try:
            full_fn = os.path.join(src, fn + ".json")
            with open(full_fn, "r") as f:
                lbl_json = json.load(f)
            
            pdf_file = os.path.join(src, fn + ".pdf")
            images = convert_from_path(pdf_file)

            # make folder_specific_folder
            save_curr_image_path = os.path.join(tr_save_split_img, fn)
            save_curr_label_path = os.path.join(tr_save_split_lbl, fn)

            os.makedirs(save_curr_image_path, exist_ok=True)
            os.makedirs(save_curr_label_path, exist_ok=True)

            for img, page in zip(images, lbl_json['pages']):
                img = np.asarray(img)
                h,w,_ = img.shape
                for curr_idx, (box, txt) in enumerate(zip(page['bbox'], page['text'])):
                    left, top, width, height = box
                    crop = img[
                        int(round(top*h)):int(round(top*h+height*h)),
                        int(round(left*w)):int(round(left*w+width*w))
                    ]
                    Image.fromarray(crop).save(os.path.join(save_curr_image_path, f"{str(curr_idx).zfill(6)}.png"))
                    with open(os.path.join(save_curr_label_path, f"{str(curr_idx).zfill(6)}.txt"), "w") as lblf:
                        lblf.write(txt)
        except Exception as e:
            print("Ran into issue:")
            traceback.print_exc()
            print("Skipping..")
    for fn in tqdm(val_list):
        try:
            full_fn = os.path.join(src, fn + ".json")
            with open(full_fn, "r") as f:
                lbl_json = json.load(f)
            
            pdf_file = os.path.join(src, fn + ".pdf")
            images = convert_from_path(pdf_file)

            # make folder_specific_folder
            save_curr_image_path = os.path.join(vl_save_split_img, fn)
            save_curr_label_path = os.path.join(vl_save_split_lbl, fn)

            os.makedirs(save_curr_image_path, exist_ok=True)
            os.makedirs(save_curr_label_path, exist_ok=True)

            for img, page in zip(images, lbl_json['pages']):
                img = np.asarray(img)
                h,w,_ = img.shape
                for curr_idx, (box, txt) in enumerate(zip(page['bbox'], page['text'])):
                    left, top, width, height = box
                    crop = img[
                        int(round(top*h)):int(round(top*h+height*h)),
                        int(round(left*w)):int(round(left*w+width*w))
                    ]
                    Image.fromarray(crop).save(os.path.join(save_curr_image_path, f"{str(curr_idx).zfill(6)}.png"))
                    with open(os.path.join(save_curr_label_path, f"{str(curr_idx).zfill(6)}.txt"), "w") as lblf:
                        lblf.write(txt)
        except Exception as e:
            print("Ran into issue:")
            traceback.print_exc()
            print("Skipping..")