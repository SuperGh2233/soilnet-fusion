import os

def get_ids(folder):
    return set([f.split('_')[0] for f in os.listdir(folder) if f.endswith('.tif')])

train_ids = get_ids('dataset/l8_images_CN/train/')
val_ids = get_ids('dataset/l8_images_CN/val/')
test_ids = get_ids('dataset/l8_images_CN/test/')

print('train/val overlap:', train_ids & val_ids)
print('train/test overlap:', train_ids & test_ids)
print('val/test overlap:', val_ids & test_ids)