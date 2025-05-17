# move the images to different dir (originally all in dataset, move them to images, masks, train, test)

import os
import shutil
from glob import glob

base_dir = "dataset"
categories = ["train", "test"]

for split in ["images", "masks"]:
    for cat in categories:
        os.makedirs(os.path.join(base_dir, split, cat), exist_ok=True)

# images
image_files = glob(os.path.join(base_dir, "*.bmp"))
for img_path in image_files:
    filename = os.path.basename(img_path)
    if filename.endswith("anno.bmp"):
        continue
    if filename.startswith("train_"):
        dest = os.path.join(base_dir, "images", "train", filename)
    elif filename.startswith("test"):
        dest = os.path.join(base_dir, "images", "test", filename)
    else:
        continue
    shutil.move(img_path, dest)

# masks
mask_files = glob(os.path.join(base_dir, "*.bmp"))
for mask_path in mask_files:
    filename = os.path.basename(mask_path)
    if filename.startswith("train_"):
        dest = os.path.join(base_dir, "masks", "train", filename)
    elif filename.startswith("test"):
        dest = os.path.join(base_dir, "masks", "test", filename)
    else:
        continue
    shutil.move(mask_path, dest)

print("move img to dir done")
