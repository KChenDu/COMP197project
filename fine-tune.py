# from data.loader import H5ImageLoader # This approach will probably not be used
from torchvision.datasets import OxfordIIITPet
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
from torch.utils.data import DataLoader
from models.nn import PetModel
from torch.optim import AdamW
from utils import Trainer


if __name__ == '__main__':
    batch_size = 16

    # Download the dataset if it doesn't exist
    OxfordIIITPet('data', download=True)

    # Load the train and validation datasets
    train_dataset = SimpleOxfordPetDataset("data/oxford-iiit-pet", "train")
    valid_dataset = SimpleOxfordPetDataset("data/oxford-iiit-pet", "valid")

    # It is a good practice to check datasets don't intersect with each other
    assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames))

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size)

    # Using professor's Dataloader (This approach will probably not be used)
    # train_dataloader = H5ImageLoader('data/oxford-iiit-pet/images_train.h5', 1, 'data/oxford-iiit-pet/labels_train.h5')
    # valid_dataloader = H5ImageLoader('data/oxford-iiit-pet/images_val.h5', 20, 'data/oxford-iiit-pet/labels_val.h5')

    # import the model
    model = PetModel()
    # define the optimizer
    optimizer = AdamW(model.parameters())

    trainer = Trainer()
    trainer.fit(model, train_dataloader, valid_dataloader, optimizer)
