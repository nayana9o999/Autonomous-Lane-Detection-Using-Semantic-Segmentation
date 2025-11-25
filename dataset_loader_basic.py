import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

BASE_DIR = "/content/IndianRoadSeg/Indian_road_data/Indian_road_data"
IMAGE_DIR = os.path.join(BASE_DIR, "Raw_images")
MASK_DIR = os.path.join(BASE_DIR, "Masks")

IMG_SIZE = (256, 256)

def load_dataset(split="train"):
    images, masks = [], []
    seq_dirs = glob(os.path.join(IMAGE_DIR, split, "*"))

    for seq in tqdm(seq_dirs, desc=f"Loading {split} data"):
        seq_name = os.path.basename(seq)
        img_files = sorted(glob(os.path.join(seq, "*.jpg")))

        mask_seq_dir = os.path.join(MASK_DIR, split, seq_name)

        for img_path in img_files:
            frame_id = os.path.basename(img_path).replace("_leftImg8bit.jpg", "")
            mask_path = os.path.join(mask_seq_dir, f"{frame_id}_gtFine_labelTrainIds.png")

            if not os.path.exists(mask_path):
                continue

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            img = cv2.resize(img, IMG_SIZE)
            mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

            img = img.astype(np.float32) / 255.0
            mask = mask.astype(np.int32)

            images.append(img)
            masks.append(mask)

    return np.array(images), np.array(masks)
