from settings import (SEED,
                      OxfordIIITPet_DATA_ROOT,
                      FINE_TUNING_BATCH_SIZE as BATCH_SIZE,
                      N_CPU,
                      MODEL,
                      FINE_TUNING_OPTIMIZER as OPTIMIZER,
                      FINE_TUNING_MAX_EPOCHS as MAX_EPOCHS,
                      FINE_TUNING_FREQ_INFO as FREQ_INFO,
                      FINE_TUNING_FREQ_SAVE as FREQ_SAVE)
from torch import manual_seed
from data.datasets import SimpleOxfordPetDataset
from torch.utils.data import DataLoader
from utils import Trainer


if __name__ == '__main__':
    manual_seed(SEED)

    # Download the dataset if it does not exist
    SimpleOxfordPetDataset.download(OxfordIIITPet_DATA_ROOT)

    # Load the train and validation datasets
    train_dataset = SimpleOxfordPetDataset(OxfordIIITPet_DATA_ROOT, "train")
    valid_dataset = SimpleOxfordPetDataset(OxfordIIITPet_DATA_ROOT, "valid")

    # It is a good practice to check datasets don't intersect with each other
    assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames))

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=N_CPU)
    valid_dataloader = DataLoader(valid_dataset, BATCH_SIZE, num_workers=N_CPU)

    # Import the model
    model = MODEL()
    # Define the optimizer
    optimizer = OPTIMIZER(model.parameters())

    trainer = Trainer(MAX_EPOCHS, FREQ_INFO, FREQ_SAVE)
    trainer.fit(model, train_dataloader, valid_dataloader, optimizer)
