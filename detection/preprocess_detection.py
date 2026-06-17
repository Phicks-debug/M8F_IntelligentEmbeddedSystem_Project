import os

root = r"C:\Users\jules\Documents\prog\M8F_IntelligentEmbeddedSystem_Project\detection\data\raw"

# Datasets
bark_path = os.path.join(root, "tree-bark")
mushroom_path = os.path.join(root, "mushroom")
flowers_path = os.path.join(root, "flowers")
lumber_path = os.path.join(root, "timber")

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
root = "C:\\Users\\jules\\Documents\\prog\\M8F_IntelligentEmbeddedSystem_Project\\detection\\data"

os.makedirs(os.path.join(root, "train"), exist_ok=True)
os.makedirs(os.path.join(root, "val"), exist_ok=True)
os.makedirs(os.path.join(root, "test"), exist_ok=True)

os.makedirs(os.path.join(root, "train", "no-mushroom"), exist_ok=True)
os.makedirs(os.path.join(root, "train", "mushroom"), exist_ok=True)
os.makedirs(os.path.join(root, "val", "no-mushroom"), exist_ok=True)
os.makedirs(os.path.join(root, "val", "mushroom"), exist_ok=True)
os.makedirs(os.path.join(root, "test", "no-mushroom"), exist_ok=True)
os.makedirs(os.path.join(root, "test", "mushroom"), exist_ok=True)


print("division")

# Split the datasets into train, validation, and test sets
import random
import shutil

random.seed(42)

for dataset_path, image_list in dataset_lists.items():
    random.shuffle(image_list)
    train_size = int(0.8 * len(image_list))
    val_size = int(0.10 * len(image_list))
    test_size = len(image_list) - train_size - val_size

    train_images = image_list[:train_size]
    val_images = image_list[train_size:train_size + val_size]
    test_images = image_list[train_size + val_size:train_size + val_size + test_size]

    # Copy images to respective directories
    for img in train_images:
        if dataset_path != mushroom_path:
            os.makedirs(os.path.join(root, "train", "no-mushroom"), exist_ok=True)
            shutil.copy(img, os.path.join(root, "train", "no-mushroom"))
        else:
            os.makedirs(os.path.join(root, "train", "mushroom"), exist_ok=True)
            shutil.copy(img, os.path.join(root, "train", "mushroom"))

    for img in val_images:
        if dataset_path != mushroom_path:
            os.makedirs(os.path.join(root, "val", "no-mushroom"), exist_ok=True)
            shutil.copy(img, os.path.join(root, "val", "no-mushroom"))
        else:
            os.makedirs(os.path.join(root, "val", "mushroom"), exist_ok=True)
            shutil.copy(img, os.path.join(root, "val", "mushroom"))

    for img in test_images:
        if dataset_path != mushroom_path:
            os.makedirs(os.path.join(root, "test", "no-mushroom"), exist_ok=True)
            shutil.copy(img, os.path.join(root, "test", "no-mushroom"))
        else:
            os.makedirs(os.path.join(root, "test", "mushroom"), exist_ok=True)
            shutil.copy(img, os.path.join(root, "test", "mushroom"))