from settings import (SEED,
                      DATA_ROOT,
                      FINE_TUNING_TRANSFORMS as TRANSFORMS,
                      FINE_TUNING_BATCH_SIZE as BATCH_SIZE,
                      N_CPU,
                      FINE_TUNING_MODEL as MODEL,
                      FINE_TUNING_OPTIMIZER as OPTIMIZER,
                      FINE_TUNING_MAX_EPOCHS as MAX_EPOCHS,
                      FINE_TUNING_FREQ_INFO as FREQ_INFO,
                      FINE_TUNING_FREQ_SAVE as FREQ_SAVE,
                      DEVICE)
from torch import manual_seed
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import random_split, DataLoader
from utils import FineTuner


if __name__ == '__main__':
    manual_seed(SEED)

    # Load the train and validation datasets
    train_dataset, valid_dataset = random_split(OxfordIIITPet(DATA_ROOT, target_types='segmentation', transforms=TRANSFORMS, download=True), (.9, .1))

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=N_CPU)
    valid_dataloader = DataLoader(valid_dataset, BATCH_SIZE, num_workers=N_CPU)

    # Import the model
    model = MODEL()
    # Define the optimizer
    optimizer = OPTIMIZER(model.parameters())

    finetuner = FineTuner(MAX_EPOCHS, FREQ_INFO, FREQ_SAVE, DEVICE)
    finetuner.fit(model, train_dataloader, valid_dataloader, optimizer)
