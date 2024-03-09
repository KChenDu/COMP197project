from settings import (SEED,
                      DATA_ROOT,
                      TRANSFORM,
                      FINE_TUNING_BATCH_SIZE as BATCH_SIZE,
                      N_CPU,
                      MODEL,
                      FINE_TUNING_OPTIMIZER as OPTIMIZER,
                      FINE_TUNING_MAX_EPOCHS as MAX_EPOCHS,
                      FINE_TUNING_FREQ_INFO as FREQ_INFO,
                      FINE_TUNING_FREQ_SAVE as FREQ_SAVE)
from torch import manual_seed
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import random_split, DataLoader
from utils import Trainer


if __name__ == '__main__':
    manual_seed(SEED)

    # Download and load the trainval dataset
    dataset = OxfordIIITPet(DATA_ROOT, target_types="segmentation", transform=TRANSFORM, target_transform=TRANSFORM, download=True)

    # Download and load the train and validation datasets
    train_dataset, valid_dataset = random_split(dataset, [.9, .1])

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=N_CPU)
    valid_dataloader = DataLoader(valid_dataset, BATCH_SIZE, num_workers=N_CPU)

    # Import the model
    model = MODEL()
    # Define the optimizer
    optimizer = OPTIMIZER(model.parameters())

    trainer = Trainer(MAX_EPOCHS, FREQ_INFO, FREQ_SAVE)
    trainer.fit(model, train_dataloader, valid_dataloader, optimizer)
