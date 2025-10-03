# AI Hub 한국어 글자체 데이터 전처리 - 13.한국어글자체 중 특별히 '04. Text in the wild_230209_add' 별도처리

import json
import cv2
import argparse
import os
from tqdm import tqdm
import multiprocessing as mp
from sklearn.model_selection import train_test_split
import shutil
import traceback

TRAIN_SPLIT = 0.8


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--destination", type=str, required=True)

    args = parser.parse_args()
    folder_path = args.source
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

    # for each folder process them and get train test split in flat folder
    json_file = [x for x in os.listdir(folder_path) if x.endswith("json")][0]
    with open(os.path.join(folder_path, json_file), "r") as f:
        lbl_json = json.load(f)

    # get all the folder paths
    image_paths = {}
    # use annotation file

    images_info = lbl_json['images']
    for info in images_info:
        fn = info['file_name']
        type_ = info['type']
        if type_ == "book":
            subfolder = os.path.join(folder_path, '01_textinthewild_book_images_new', 'book')
        elif type_ == "product":
            subfolder = os.path.join(folder_path, '01_textinthewild_goods_images_new', 'Goods')
        elif type_ == "sign":
            subfolder = os.path.join(folder_path, '01_textinthewild_signboard_images_new', 'Signboard')
        elif type_ == "traffic sign":
            subfolder = os.path.join(folder_path, '01_textinthewild_traffic_sign_images_new', 'Traffic_Sign')
        
        image_paths[info['id']] = os.path.join(subfolder, fn)


    annotations = lbl_json['annotations']

    train_annotations, test_annotations = train_test_split(annotations, test_size=0.2, random_state=42)

    print(f"processing {len(annotations)} files..")

    for ann in tqdm(train_annotations):
        try:
            lbl_id = ann['id']
            id_ = ann['image_id']
            txt = ann['text']
            if txt is None or txt == '':
                print(f"invalid txt label '{txt}' found, skipping..")
                continue

            if id_ not in image_paths or not os.path.isfile(image_paths[id_]):
                print(f"File {image_path} does not exist, skipping..")
                continue
            image_path = image_paths[id_]
            img = cv2.imread(image_path)
            bbox = ann['bbox']
            img = img[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
            cv2.imwrite(os.path.join(tr_save_split_img, f"{lbl_id}.png"), img)

            with open(f"{tr_save_split_lbl}/{lbl_id}.txt", "w", encoding="UTF-8") as f:
                f.write(txt)
        except Exception as e:
            print("Ran into the following issue:")
            traceback.print_exc()
            print("Skipping..")
            # make sure there isn't dangling files for skipped samples
            if os.path.isfile(os.path.join(tr_save_split_img, f"{lbl_id}.png")):
                os.remove(os.path.join(tr_save_split_img, f"{lbl_id}.png"))
            if os.path.isfile(f"{tr_save_split_lbl}/{lbl_id}.txt"):
                os.remove(f"{tr_save_split_lbl}/{lbl_id}.txt")

    for ann in tqdm(test_annotations):
        try:
            lbl_id = ann['id']
            id_ = ann['image_id']
            txt = ann['text']
            if txt is None or txt == '':
                print(f"invalid txt label '{txt}' found, skipping..")
                continue

            if id_ not in image_paths or not os.path.isfile(image_paths[id_]):
                print(f"File {image_path} does not exist, skipping..")
                continue
            image_path = image_paths[id_]
            img = cv2.imread(image_path)
            bbox = ann['bbox']
            img = img[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
            cv2.imwrite(os.path.join(tr_save_split_img, f"{lbl_id}.png"), img)

            with open(f"{vl_save_split_lbl}/{lbl_id}.txt", "w", encoding="UTF-8") as f:
                f.write(txt)
        except Exception as e:
            print("Ran into the following issue:")
            traceback.print_exc()
            print("Skipping..")
        
