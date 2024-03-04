from data.loader import H5ImageLoader
from models.nn import PetModel
from utils import Trainer


if __name__ == '__main__':
    train_dataloader = H5ImageLoader('data/oxford-iiit-pet/images_train.h5', 1, 'data/oxford-iiit-pet/labels_train.h5')
    valid_dataloader = H5ImageLoader('data/oxford-iiit-pet/images_val.h5', 20, 'data/oxford-iiit-pet/labels_val.h5')
    model = PetModel()
    trainer = Trainer()  # There are adjustable parameters in PetModel
    optimizer = None  # Fill this
    trainer.fit(model, train_dataloader, valid_dataloader, optimizer)
