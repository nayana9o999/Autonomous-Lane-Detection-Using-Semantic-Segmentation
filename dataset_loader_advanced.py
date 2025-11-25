import os
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

DATA_DIR = "/content/IndianRoadSeg/Indian_road_data/Indian_road_data"
IMG_HEIGHT, IMG_WIDTH = 256, 256

def load_images_from_folder(image_folder, mask_folder):
    images, masks = [], []
    for subdir in sorted(os.listdir(image_folder)):
        img_subdir = os.path.join(image_folder, subdir)
        mask_subdir = os.path.join(mask_folder, subdir)
        if not os.path.isdir(img_subdir):
            continue

        print(f"📁 Scanning folder: {subdir}")
        for img_name in tqdm(os.listdir(img_subdir), desc=f"Loading {subdir}", leave=False):
            if not img_name.endswith("_leftImg8bit.jpg"):
                continue

            frame_id = img_name.replace("_leftImg8bit.jpg", "")
            mask_name = f"{frame_id}_gtFine_labelTrainIds.png"
            mask_path = os.path.join(mask_subdir, mask_name)

            if not os.path.exists(mask_path):
                continue

            img_path = os.path.join(img_subdir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img.astype(np.float32) / 255.0

            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)

            images.append(img)
            masks.append(mask)

    return np.array(images), np.array(masks)
