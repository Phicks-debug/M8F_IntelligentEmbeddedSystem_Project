import os
import random
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "data" / "raw"
OUTPUT_DIR = SCRIPT_DIR / "data"

# Datasets
bark_path = RAW_DIR / "tree-bark"
mushroom_path = RAW_DIR / "mushroom"
flowers_path = RAW_DIR / "flowers"
lumber_path = RAW_DIR / "timber"

image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}

dataset_paths = [bark_path, mushroom_path, flowers_path, lumber_path]
dataset_lists = {}

for dataset_path in dataset_paths:
    print(f"Processing dataset: {dataset_path}")
    dataset_lists[dataset_path] = []

    for dirpath, _, filenames in os.walk(dataset_path):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                full_path = os.path.join(dirpath, filename)
                dataset_lists[dataset_path].append(full_path)

# Print the number of images in each dataset
for dataset_path, image_list in dataset_lists.items():
    print(f"{dataset_path}: {len(image_list)} images")


print("dirs")

# make directories
for split in ["train", "val", "test"]:
    (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / split / "no-mushroom").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / split / "mushroom").mkdir(parents=True, exist_ok=True)


print("division")

# Split the datasets into train, validation, and test sets
random.seed(42)

for dataset_path, image_list in dataset_lists.items():
    random.shuffle(image_list)
    train_size = int(0.8 * len(image_list))
    val_size = int(0.10 * len(image_list))

    train_images = image_list[:train_size]
    val_images = image_list[train_size:train_size + val_size]
    test_images = image_list[train_size + val_size:]

    # Copy images to respective directories
    for img in train_images:
        label = "mushroom" if dataset_path == mushroom_path else "no-mushroom"
        shutil.copy(img, OUTPUT_DIR / "train" / label)

    for img in val_images:
        label = "mushroom" if dataset_path == mushroom_path else "no-mushroom"
        shutil.copy(img, OUTPUT_DIR / "val" / label)

    for img in test_images:
        label = "mushroom" if dataset_path == mushroom_path else "no-mushroom"
        shutil.copy(img, OUTPUT_DIR / "test" / label)
