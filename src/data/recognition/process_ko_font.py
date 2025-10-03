# AI Hub 한국어 글자체 데이터 전처리
# example: python process_ko_font.py --source 13.한국어글자체 --destination ko-font-processed


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


    for folder in os.listdir(src):
        folder_path = os.path.join(src, folder)
        if not os.path.isdir(folder_path) or folder.split(".")[0] not in ['01', '02', '03']:
            continue

        # get all the folder paths
        image_paths = {}
        for subfolder in os.listdir(folder_path):
            print(subfolder)
            subfolder_path = os.path.join(folder_path, subfolder)
            if os.path.isdir(subfolder_path):
                for im in [x for x in os.listdir(subfolder_path) if x.endswith(".png") or x.endswith(".jpg")]:
                    image_paths[im.split(".")[0]] = os.path.join(subfolder_path, im)

        # for each folder process them and get train test split in flat folder
        json_file = [x for x in os.listdir(folder_path) if x.endswith("json")][0]
        with open(os.path.join(folder_path, json_file), "r") as f:
            lbl_json = json.load(f)
            # info, images, annotations, licenses

        annotations = lbl_json['annotations']

        train_annotations, test_annotations = train_test_split(annotations, test_size=0.2, random_state=42)

        print(f"processing {len(annotations)} files..")

        for ann in tqdm(train_annotations):
            id_ = ann['image_id']
            txt = ann['text']
            if id_ not in image_paths or not os.path.isfile(image_paths[id_]):
                print(f"File {image_path} does not exist, skipping..")
                continue
            image_path = image_paths[id_]

            shutil.copyfile(image_path, os.path.join(tr_save_split_img, f"{id_}.png"))
            with open(f"{tr_save_split_lbl}/{id_}.txt", "w", encoding="UTF-8") as f:
                f.write(txt)

        for ann in tqdm(test_annotations):
            id_ = ann['image_id']
            txt = ann['text']
            if id_ not in image_paths or not os.path.isfile(image_paths[id_]):
                print(f"File {image_path} does not exist, skipping..")
                continue
            image_path = image_paths[id_]

            shutil.copyfile(os.path.join(image_path), os.path.join(vl_save_split_img, f"{id_}.png"))
            with open(f"{vl_save_split_lbl}/{id_}.txt", "w", encoding="UTF-8") as f:
                f.write(txt)
