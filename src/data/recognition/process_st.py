# Synthtiger v1.1 전처리
# example: python process_st.py --source synthtiger_v1.1 --destination synthtiger_v1.1-processed

import json
import cv2
import argparse
import os
from tqdm import tqdm
import multiprocessing as mp
from sklearn.model_selection import train_test_split
import shutil

TRAIN_SPLIT = 0.8


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--destination", type=str, required=True)

    args = parser.parse_args()
    src = args.source
    dest = args.destination

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

    with open(f"{src}/gt.txt", "r") as f:
        annotations = f.readlines()

    train_annotations, test_annotations = train_test_split(annotations, test_size=0.2, random_state=42)
    print(f"processing {len(annotations)} files..")

    for ann in tqdm(train_annotations):
        im_path, txt = ann.split()
        im_name = im_path.split("/")[-1]
        if not os.path.isfile(os.path.join(src,im_path)):
            print(f"File {im_path} does not exist, skipping..")
            continue

        shutil.copyfile(os.path.join(src,im_path), os.path.join(tr_save_split_img, im_name))
        lbl_name = im_name.split(".")[0]
        with open(f"{tr_save_split_lbl}/{lbl_name}.txt", "w", encoding="UTF-8") as f:
            f.write(txt)

    for ann in tqdm(test_annotations):
        im_path, txt = ann.split()
        im_name = im_path.split("/")[-1]
        if not os.path.isfile(os.path.join(src,im_path)):
            print(f"File {im_path} does not exist, skipping..")
            continue

        shutil.copyfile(os.path.join(src,im_path), os.path.join(vl_save_split_img, im_name))
        lbl_name = im_name.split(".")[0]
        with open(f"{vl_save_split_lbl}/{lbl_name}.txt", "w", encoding="UTF-8") as f:
            f.write(txt)
