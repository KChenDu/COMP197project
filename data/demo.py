import numpy as np

from os import path
from PIL import Image
from loader import H5ImageLoader

DATA_PATH = path.dirname(path.realpath(__file__)) + '/oxford-iiit-pet'

images,labels = next(iter(H5ImageLoader(DATA_PATH+'/images_train.h5', 10, DATA_PATH+'/labels_train.h5')))
image_montage = Image.fromarray(np.concatenate([images[i] for i in range(len(images))],axis=1))
image_montage.save("train_images.jpg")
label_montage = Image.fromarray(np.concatenate([labels[i] for i in range(len(labels))],axis=1))
label_montage.save("train_labels.jpg")
