import os, random, shutil

img_dir = 'dataset/l8_images_CN/'
files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
random.seed(42)
random.shuffle(files)
n = len(files)
train, val = int(n*0.7), int(n*0.15)
splits = [('train', files[:train]), ('val', files[train:train+val]), ('test', files[train+val:])]

for split, split_files in splits:
    split_dir = os.path.join(img_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    for f in split_files:
        shutil.move(os.path.join(img_dir, f), os.path.join(split_dir, f))