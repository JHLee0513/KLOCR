# AI Hub 공공행정문서 OCR dataset

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
        folder_path = os.path.join(src, folder)

        if folder == 'Training':
            save_split_img = tr_save_split_img
            save_split_lbl = tr_save_split_lbl
            input_folder_path = os.path.join(folder_path, '02.원천데이터(jpg)')

        elif folder == 'Validation':
            save_split_img = vl_save_split_img
            save_split_lbl = vl_save_split_lbl
            input_folder_path = os.path.join(folder_path, '02.원천데이터(Jpg)')

        # input
        # label_folder_path = os.path.join(folder_path, '01.라벨링데이터(Json)')

        # get all the folder paths
        image_paths = {}
        for topic in tqdm(os.listdir(input_folder_path)):
            topic_path = os.path.join(input_folder_path, topic)

            for subset in os.listdir(topic_path):
                subset_path = os.path.join(topic_path, subset)
                for sample_folder in os.listdir(subset_path):
                    sample_folder_path = os.path.join(subset_path, sample_folder)
                    for sample in os.listdir(sample_folder_path):
                        sample_img_path = os.path.join(sample_folder_path, sample)
                        sample_name = sample.replace(".jpg", "")
                        if os.path.isdir(os.path.join(save_split_img, sample_name)):
                            continue
                        try:
                            image = cv2.imread(sample_img_path)
                            if image.shape[0] < 10 or image.shape[1] < 10:
                                print('Detected Image does not meet minimum size')
                                continue
                        except Exception as e:
                            print(f"Error loading image {sample}")
                            continue


                        os.makedirs(os.path.join(save_split_img, sample_name), exist_ok=True)
                        os.makedirs(os.path.join(save_split_lbl, sample_name), exist_ok=True)
                        if folder == 'Validation':
                            sample_lbl_path = sample_img_path.replace("02.원천데이터(Jpg)", "01.라벨링데이터(Json)").replace(".jpg", ".json")
                        else:
                            sample_lbl_path = sample_img_path.replace("02.원천데이터(jpg)", "01.라벨링데이터(Json)").replace(".jpg", ".json")
                        try:
                            with open (sample_lbl_path, "r") as f:
                                lbl = json.load(f)
                        except Exception as e:
                            print(f"Error reading label json: {e}")
                            continue
                        for i, box in enumerate(lbl['annotations']):
                            txt = box['annotation.text']
                            box_coord = box['annotation.bbox']
                            crop = image[box_coord[1]:box_coord[1] + int(box_coord[3]), box_coord[0]:box_coord[0] + int(box_coord[2])]
                            if crop.sum() <= 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                                # print("crop issue!")
                                continue
                            # name = sample.replace(".jpg", "") + f"_{str(i).zfill(6)}"
                            name = f"{str(i).zfill(6)}"
                            cv2.imwrite(f"{save_split_img}/{sample_name}/{name}.jpg", crop)
                            with open(f"{save_split_lbl}/{sample_name}/{name}.txt", "w", encoding="UTF-8") as f:
                                f.write(txt)
                            
