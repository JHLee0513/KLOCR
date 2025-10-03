# https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=ocr&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71301
# example: python process_ko_ocr_log.py --source ko-ocr-logistics --destination ko-ocr-logistics-processed

import yaml
import cv2
import argparse
import os
from tqdm import tqdm
import multiprocessing as mp

def save_crop(data):
    i, json, split_folder_path, save_split_img, save_split_lbl = data
    if not json.endswith(".json"):
        print("skipping file that's not valid JSON annotation..")
        return
    json_path = os.path.join(split_folder_path, json)
    # get image path
    image_path = json_path.replace("02.라벨링데이터", "01.원천데이터").replace(".json", ".png")
    with open(json_path, "rb") as f:
        label = yaml.safe_load(f)
    # is_finance_data = '물류' not in label['Dataset']['name']
    try:
        image = cv2.imread(image_path)
    except Exception as e:
        print(f'Detected Image error for {json_path} : {e}')
        return
    if image is None:
        print(f'Detected Image error for {json_path} : image is None')
        return
    elif image.shape[0] < 10 or image.shape[1] < 10:
        print('Detected Image does not meet minimum size')
        return
    
    for i, box in enumerate(label['bbox']):
        txt = box['data']
        x = box['x']
        y = box['y']
        crop = image[min(y):max(y),min(x):max(x)]
        if crop.sum() <= 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            # print("crop issue!")
            continue
        # try:
        name = json.replace(".json", "") + f"_{str(i).zfill(6)}"
        cv2.imwrite(f"{save_split_img}/{name}.png", crop)
        with open(f"{save_split_lbl}/{name}.txt", "w", encoding="UTF-8") as f:
            f.write(txt)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--destination", type=str, required=True)

    args = parser.parse_args()

    dest = args.destination

    for split in ['Training', 'Validation']:
        print(f"Processing split {split}...")
        split_folder_path = os.path.join(args.source, split, "02.라벨링데이터")
        save_split_path = os.path.join(dest, split)
        save_split_img = os.path.join(save_split_path, "images")
        save_split_lbl = os.path.join(save_split_path, "label")

        os.makedirs(save_split_path, exist_ok=True)
        os.makedirs(save_split_img, exist_ok=True)
        os.makedirs(save_split_lbl, exist_ok=True)

        tasks = [(i,x, split_folder_path, save_split_img, save_split_lbl) for i,x in enumerate(os.listdir(split_folder_path))]

        # tasks = enumerate(tqdm([os.path.join(split_folder_path, x) for x in os.listdir(split_folder_path)], total=len(os.listdir(split_folder_path))))
        pool = mp.Pool(processes=16)
        pool.map(save_crop, tasks)
        pool.close()
        pool.join()

        # for i, json in tqdm(enumerate(os.listdir(split_folder_path)), total=len(os.listdir(split_folder_path))):
        #     if not json.endswith(".json"):
        #         print("skipping file that's not valid JSON annotation..")
        #         continue
        #     json_path = os.path.join(split_folder_path, json)
        #     # get image path
        #     image_path = json_path.replace("02.라벨링데이터", "01.원천데이터").replace(".json", ".png")
        #     with open(json_path, "rb") as f:
        #         label = yaml.safe_load(f)
        #     # is_finance_data = '물류' not in label['Dataset']['name']
        #     try:
        #         image = cv2.imread(image_path)
        #     except Exception as e:
        #         print(f'Detected Image error for {json_path} : {e}')
        #         continue
        #     if image is None:
        #         print(f'Detected Image error for {json_path} : image is None')
        #         continue
        #     elif image.shape[0] < 10 or image.shape[1] < 10:
        #         print('Detected Image does not meet minimum size')
        #         continue

        #     tasks = zip(range(len(label['bbox'])), label['bbox'])
        #     pool = mp.Pool(processes=16)
        #     pool.map(save_crop, tasks)
        #     pool.close()
        #     pool.join()