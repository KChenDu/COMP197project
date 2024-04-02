from functools import partial
import torch
from models.mae import MaskedAutoencoderViT
from settings import (SEED,
                      DEVICE,
                      setup_device,
                      DATA_ROOT,
                      PRE_TRAINING_TRANSFORM as TRANSFORM,
                      PRE_TRAINING_BATCH_SIZE as BATCH_SIZE,
                      NUM_WORKERS,
                      PIN_MEMORY,
                      PRE_TRAINING_MODEL as MODEL,
                      PRE_TRAINING_OPTIMIZER as OPTIMIZER,
                      PRE_TRAINING_MAX_EPOCHS as MAX_EPOCHS,
                      PRE_TRAINING_FREQ_INFO as FREQ_INFO,
                      PRE_TRAINING_FREQ_SAVE as FREQ_SAVE,
                      MASK_RATIO,
                      LR_SCHED_ARGS)
from torch import manual_seed, Generator
import torch.nn as nn
from torchvision.datasets import ImageNet
from data.datasets import KaggleCatsAndDogsDataset
from torch.utils.data import DataLoader
from utils import PreTrainer


if __name__ == '__main__':
    manual_seed(SEED)
    setup_device()

    # Load the train dataset
    train_dataset = KaggleCatsAndDogsDataset(DATA_ROOT, transform=TRANSFORM, num_samples=1000)
    # train_dataset = KaggleCatsAndDogsDataset(DATA_ROOT, transform=TRANSFORM)

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, generator=Generator(device=DEVICE), pin_memory=PIN_MEMORY)

    # Import the model
    model = MODEL(img_size=224,
                    patch_size=16,
                    in_chans=3,
                    out_chans=[512, 320, 128, 64],
                    embed_dim=768,
                    depth=5,
                    decoder_embed_dim=512,
                    num_heads=12,
                    mlp_ratio=4)
    # model = MODEL()
    
    # Define the optimizer
    optimizer = OPTIMIZER(model.parameters())

    pretrainer = PreTrainer(MAX_EPOCHS, FREQ_INFO, FREQ_SAVE, DEVICE)
    pretrainer.fit(model, train_dataloader, optimizer, MASK_RATIO, LR_SCHED_ARGS)

    torch.save(model.state_dict(), 'model_pre_trained.pth')
