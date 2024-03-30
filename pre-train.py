from settings import (SEED,
                      DATA_ROOT,
                      PRE_TRAINING_TRANSFORM as TRANSFORM,
                      PRE_TRAINING_BATCH_SIZE as BATCH_SIZE,
                      N_CPU,
                      PRE_TRAINING_MODEL as MODEL,
                      PRE_TRAINING_OPTIMIZER as OPTIMIZER,
                      PRE_TRAINING_MAX_EPOCHS as MAX_EPOCHS,
                      PRE_TRAINING_FREQ_INFO as FREQ_INFO,
                      PRE_TRAINING_FREQ_SAVE as FREQ_SAVE,
                      DEVICE,
                      MASK_RATIO)
from torch import manual_seed
from torchvision.datasets import ImageNet
from data.datasets import KaggleCatsAndDogsDataset
from torch.utils.data import DataLoader
from utils import PreTrainer


if __name__ == '__main__':
    manual_seed(SEED)

    # Load the train dataset
    # train_dataset = ImageNet(DATA_ROOT, 'val', transform=TRANSFORM)
    train_dataset = KaggleCatsAndDogsDataset(DATA_ROOT, transform=TRANSFORM)

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=N_CPU)

    # Import the model
    model = MODEL()
    # Define the optimizer
    optimizer = OPTIMIZER(model.parameters())

    pretrainer = PreTrainer(MAX_EPOCHS, FREQ_INFO, FREQ_SAVE, DEVICE)
    pretrainer.fit(model, train_dataloader, optimizer, MASK_RATIO)
