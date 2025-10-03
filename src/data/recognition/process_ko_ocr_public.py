# AI Hub OCR 데이터 (공공) processing
# https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71299

import json
import cv2
import argparse
import os
from tqdm import tqdm

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
        if folder not in ['Training', 'Validation']:
            continue
        if folder == 'Training':
            save_split_img = tr_save_split_img
            save_split_lbl = tr_save_split_lbl
        elif folder == 'Validation':
            save_split_img = vl_save_split_img
            save_split_lbl = vl_save_split_lbl

        folder_path = os.path.join(src, folder)
        # input
        input_folder_path = os.path.join(folder_path, '01.원천데이터')
        # label
        label_folder_path = os.path.join(folder_path, '02.라벨링데이터')

        # get all the folder paths
        image_paths = {}
        for sample in tqdm(os.listdir(input_folder_path)):
            sample_name = sample.replace(".jpg", "")
            os.makedirs(os.path.join(save_split_img, sample_name), exist_ok=True)
            os.makedirs(os.path.join(save_split_lbl, sample_name), exist_ok=True)

            sample_img_path = os.path.join(input_folder_path, sample)
            
            try:
                image = cv2.imread(sample_img_path)
                if image.shape[0] < 10 or image.shape[1] < 10:
                    print('Detected Image does not meet minimum size')
                    continue
            except Exception as e:
                print(f"Error loading image {sample}")
                continue

            sample_lbl_path = os.path.join(label_folder_path, sample.replace(".jpg", ".json"))

            with open (sample_lbl_path, "r") as f:
                try:
                    lbl = json.load(f)
                except Exception as e:
                    print(f"Error reading label json: {e}")
                    continue
                for i, box in enumerate(lbl['Bbox']):
                    txt = box['data']
                    x = box['x']
                    y = box['y']
                    # xy are in order of top left, top right, bottom left, bottom right
                    crop = crop = image[min(y):max(y),min(x):max(x)]
                    if crop.sum() <= 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                        # print("crop issue!")
                        continue
                    # name = sample.replace(".jpg", "") + f"_{str(i).zfill(6)}"
                    name = f"{str(i).zfill(6)}"
                    cv2.imwrite(f"{save_split_img}/{sample_name}/{name}.jpg", crop)
                    with open(f"{save_split_lbl}/{sample_name}/{name}.txt", "w", encoding="UTF-8") as f:
                        f.write(txt)
