import os
import shutil
import random
import cv2
from PIL import Image
import imagehash
import numpy as np

# Paths
source_images = 'OID/Dataset/train/Car'
source_labels = 'OID/Dataset/train/Car/Label'
output_root = 'Dataset'  # Will create Dataset/train, val, test

# Create YOLO structure
for split in ['train', 'val', 'test']:
    os.makedirs(f'{output_root}/{split}/images', exist_ok=True)
    os.makedirs(f'{output_root}/{split}/labels', exist_ok=True)

# Get all images
images = [f for f in os.listdir(source_images) if f.lower().endswith(('.jpg', '.jpeg'))]

# Basic duplicate removal using perceptual hash
hash_dict = {}
duplicates = []
valid_images = []

print("Checking for duplicates and unlabeled images...")
for img_name in images:
    img_path = os.path.join(source_images, img_name)
    try:
        hash_val = imagehash.average_hash(Image.open(img_path))
        if hash_val in hash_dict:
            duplicates.append(img_name)
            print(f"Duplicate found: {img_name}")
            continue
        hash_dict[hash_val] = img_name
    except Exception as e:
        print(f"Corrupted image skipped: {img_name}")
        continue

    # Check if label exists and is not empty
    label_name = img_name.replace('.jpg', '.txt').replace('.jpeg', '.txt')
    label_path = os.path.join(source_labels, label_name)
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        print(f"No valid label: {img_name}")
        continue

    valid_images.append(img_name)

print(f"Found {len(valid_images)} valid unique images (removed {len(images) - len(valid_images)} bad ones)")

# Shuffle and split: 1100 train, 300 val, 100 test
random.shuffle(valid_images)
train_imgs = valid_images[:1100]
val_imgs = valid_images[1100:1400]
test_imgs = valid_images[1400:1500]  # up to 100

print(f"Splitting: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

# Function to convert OID bbox to YOLO format (normalized)
def convert_to_yolo(img_path, src_label_path, dst_label_path):
    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    with open(src_label_path, 'r') as f_in, open(dst_label_path, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()
            if parts[0].lower() != 'car':  # Safety check
                continue
            xmin = float(parts[1])
            ymin = float(parts[2])
            xmax = float(parts[3])
            ymax = float(parts[4])

            # YOLO format: class_id center_x center_y width height (normalized)
            x_center = (xmin + xmax) / 2 / w
            y_center = (ymin + ymax) / 2 / h
            bbox_w = (xmax - xmin) / w
            bbox_h = (ymax - ymin) / h

            f_out.write(f"0 {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")

# Copy and convert
def copy_and_convert(img_list, split_name):
    for img_name in img_list:
        src_img = os.path.join(source_images, img_name)
        dst_img = f'{output_root}/{split_name}/images/{img_name}'
        shutil.copy(src_img, dst_img)

        label_name = img_name.replace('.jpg', '.txt').replace('.jpeg', '.txt')
        src_label = os.path.join(source_labels, label_name)
        dst_label = f'{output_root}/{split_name}/labels/{label_name}'
        convert_to_yolo(src_img, src_label, dst_label)

copy_and_convert(train_imgs, 'train')
copy_and_convert(val_imgs, 'val')
copy_and_convert(test_imgs, 'test')

print("Dataset split completed!")
print("Folders created:")
print("  Dataset/train/images + labels")
print("  Dataset/val/images + labels")
print("  Dataset/test/images + labels")