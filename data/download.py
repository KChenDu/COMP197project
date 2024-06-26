
# https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
# https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

import os
import requests
import tarfile
import shutil
import random
import numpy as np
import h5py

from os import path
from PIL import Image


DATA_PATH = path.dirname(path.realpath(__file__)) + '/oxford-iiit-pet'

## download
filenames = ['images.tar.gz', 'annotations.tar.gz']
url_base = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/'

if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)
os.makedirs(DATA_PATH)

print('Downloading and extracting data...')
for temp_file in filenames:
    url = url_base + temp_file
    print(url + ' ...')
    r = requests.get(url,allow_redirects=True)
    _ = open(temp_file,'wb').write(r.content)
    with tarfile.open(temp_file) as tar_obj:
        tar_obj.extractall()
        tar_obj.close()
    os.remove(temp_file)


## spliting and converting
img_dir = 'images'
seg_dir = 'annotations/trimaps'
#----- options -----
im_size = (64,64)
ratio_val = 0.1
ratio_test = 0.2
#-------------------
img_h5s, seg_h5s = [], []
for s in ["train", "val", "test"]:
    img_h5s.append(h5py.File(os.path.join(DATA_PATH,"images_{:s}.h5".format(s)), "w"))
    seg_h5s.append(h5py.File(os.path.join(DATA_PATH,"labels_{:s}.h5".format(s)), "w"))

img_filenames = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
num_data = len(img_filenames)
num_val = int(num_data * ratio_val)
num_test = int(num_data * ratio_test)
num_train = num_data - num_val - num_test

print("Extracting data into %d-%d-%d for train-val-test (%0.2f-%0.2f-%0.2f)..." % (num_train,num_val,num_test, 1-ratio_val-ratio_test,ratio_val,ratio_test))

random.seed(90)
random.shuffle(img_filenames)

# write all images/labels to h5 file
for idx, im_file in enumerate(img_filenames):

    if idx < num_train:  # train
        ids = 0
    elif idx < (num_train + num_val):  # val
        ids = 1
    else:  # test
        ids = 2

    with Image.open(os.path.join(img_dir,im_file)) as img:
        img = np.array(img.convert('RGB').resize(im_size).getdata(),dtype='uint8').reshape(im_size[0],im_size[1],3)
        img_h5s[ids].create_dataset("{:06d}".format(idx), data=img)
    with Image.open(os.path.join(seg_dir,im_file.split('.')[0]+'.png')) as seg:
        seg = np.array(seg.resize(im_size).getdata(),dtype='uint8').reshape(im_size[0],im_size[1])
        seg_h5s[ids].create_dataset("{:06d}".format(idx), data=seg)

for ids in range(len(img_h5s)):
    img_h5s[ids].flush()
    img_h5s[ids].close()
    seg_h5s[ids].flush()
    seg_h5s[ids].close()

shutil.rmtree(img_dir)
shutil.rmtree(seg_dir.split('/')[0]) #remove entire annatations folder

print('Data saved in %s.' % os.path.abspath(DATA_PATH))
