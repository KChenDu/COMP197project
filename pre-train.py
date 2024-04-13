import torch

from settings import (SEED,
                      DEVICE,
                      DATASET,
                      DATA_ROOT,
                      PRE_TRAINING_DATA_USAGE,
                      PRE_TRAINING_TRANSFORM as TRANSFORM,
                      PRE_TRAINING_BATCH_SIZE as BATCH_SIZE,
                      PRE_TRAINING_MODEL as MODEL,
                      PRE_TRAINING_OPTIMIZER as OPTIMIZER,
                      PRE_TRAINING_MAX_EPOCHS as MAX_EPOCHS,
                      PRE_TRAINING_FREQ_INFO as FREQ_INFO,
                      PRE_TRAINING_FREQ_SAVE as FREQ_SAVE,
                      MASK_RATIO,
                      LR_SCHED_ARGS)
from torch import manual_seed, backends, Generator
from os import cpu_count
from torchvision.datasets import ImageNet
from data.datasets import KaggleCatsAndDogsDataset
from torch.utils.data import random_split, DataLoader
from utils import PreTrainer


if __name__ == '__main__':
    manual_seed(SEED)

    # Device setup
    if DEVICE == 'cuda':
        backends.cudnn.enabled = True
        torch.multiprocessing.set_start_method('spawn')
        num_workers = 0
    elif DEVICE in ['mps', 'cpu']:
        num_workers = cpu_count()
    else:
        raise ValueError

    torch.set_default_device(DEVICE)

    # Load the train dataset
    if DATASET == 'ImageNet':
        generator = Generator(DEVICE)
        train_dataset, drop = random_split(
            ImageNet(DATA_ROOT, 'val', transform=TRANSFORM),
            (PRE_TRAINING_DATA_USAGE, (1-PRE_TRAINING_DATA_USAGE)),
            generator=generator)
        
        total = len(train_dataset)+len(drop)
        print(f"trains={len(train_dataset)/total}, drop={len(drop)/total}, total={total}")

    elif DATASET == 'KaggleCatsAndDogs':
        train_dataset = KaggleCatsAndDogsDataset(DATA_ROOT, TRANSFORM)
    else:
        raise ValueError

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, generator=Generator(device=DEVICE))

    # Import the model
    model = MODEL()
    
    # Define the optimizer
    optimizer = OPTIMIZER(model.parameters())

    pretrainer = PreTrainer(MAX_EPOCHS, FREQ_INFO, FREQ_SAVE, DEVICE)
    pretrainer.fit(model, train_dataloader, optimizer, MASK_RATIO, LR_SCHED_ARGS)
