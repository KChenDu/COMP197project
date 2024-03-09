# from data.loader import H5ImageLoader # This approach will probably not be used
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import random_split, DataLoader
from models.nn import PetModel
from torch.optim import AdamW
from utils import Trainer


if __name__ == '__main__':
    batch_size = 16

    # Load the data into a train set and a validation set
    train_dataset, valid_dataset = random_split(OxfordIIITPet('data', download=True), (.9, .1))
    # Create dataloaders for both sets
    train_dataloader, valid_dataloader = DataLoader(train_dataset, batch_size, True), DataLoader(valid_dataset, batch_size)

    # This approach below will probably not be used
    # train_dataloader = H5ImageLoader('data/oxford-iiit-pet/images_train.h5', 1, 'data/oxford-iiit-pet/labels_train.h5')
    # valid_dataloader = H5ImageLoader('data/oxford-iiit-pet/images_val.h5', 20, 'data/oxford-iiit-pet/labels_val.h5')

    # import the model
    model = PetModel()
    # define the optimizer
    optimizer = AdamW(model.parameters())

    trainer = Trainer()
    trainer.fit(model, train_dataloader, valid_dataloader, optimizer)
