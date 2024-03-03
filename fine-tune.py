from data.loader import H5ImageLoader
from models import PetModel
from finetuning.utils import Trainer


train_dataloader = H5ImageLoader('data/oxford-iiit-pet/images_train.h5', 1, 'data/oxford-iiit-pet/labels_train.h5')
valid_dataloader = H5ImageLoader('data/oxford-iiit-pet/images_val.h5', 20, 'data/oxford-iiit-pet/labels_val.h5')
model = PetModel()
model
trainer = Trainer()
trainer.fit()
