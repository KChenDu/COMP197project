from settings import (SEED,
                      DATA_ROOT,
                      FINE_TUNING_TRANSFORMS as TRANSFORMS,
                      FINE_TUNING_BATCH_SIZE as BATCH_SIZE,
                      N_CPU,
                      MODEL,
                      DEVICE,
                      FINE_TUNING_OPTIMIZER as OPTIMIZER,
                      FINE_TUNING_MAX_EPOCHS as MAX_EPOCHS,
                      FINE_TUNING_FREQ_INFO as FREQ_INFO,
                      FINE_TUNING_FREQ_SAVE as FREQ_SAVE)
from torch import manual_seed, Generator
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import random_split, DataLoader
from utils import Trainer


if __name__ == '__main__':
    manual_seed(SEED)

    # Load the train and validation datasets
    train_dataset, valid_dataset = random_split(OxfordIIITPet(DATA_ROOT, target_types='segmentation', transforms=TRANSFORMS, download=True), (.9, .1))

    # Create the dataloaders
    generator = Generator(DEVICE)
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=N_CPU, generator=generator)
    valid_dataloader = DataLoader(valid_dataset, BATCH_SIZE, num_workers=N_CPU, generator=generator)

    # Import the model
    model = MODEL()
    # Define the optimizer
    optimizer = OPTIMIZER(model.parameters())

    trainer = Trainer(MAX_EPOCHS, FREQ_INFO, FREQ_SAVE)
    trainer.fit(model, train_dataloader, valid_dataloader, optimizer)
