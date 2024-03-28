from settings import (SEED,
                      SETUP_DEVICE,
                      DATA_ROOT,
                      FINE_TUNING_TRANSFORMS as TRANSFORMS,
                      FINE_TUNING_BATCH_SIZE as BATCH_SIZE,
                      DEVICE_COUNT,
                      MODEL,
                      DEVICE,
                      FINE_TUNING_OPTIMIZER as OPTIMIZER,
                      FINE_TUNING_MAX_EPOCHS as MAX_EPOCHS,
                      FINE_TUNING_FREQ_INFO as FREQ_INFO,
                      FINE_TUNING_FREQ_SAVE as FREQ_SAVE)
from torch import manual_seed, Generator, tensor, Tensor
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from utils import Trainer

import torch
import numpy as np
import PIL.Image as Image

TRAIN=True

if __name__ == '__main__':
    manual_seed(SEED)
    
    SETUP_DEVICE()
    
    generator = Generator(DEVICE)
    # Load the train and validation datasets
    train_dataset, valid_dataset = random_split(OxfordIIITPet(DATA_ROOT, target_types='segmentation', transforms=TRANSFORMS, download=True), (.9, .1), generator=generator)

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=DEVICE_COUNT, generator=generator)
    valid_dataloader = DataLoader(valid_dataset, BATCH_SIZE, num_workers=DEVICE_COUNT, generator=generator)

    # Import the model
    model = MODEL(drop_path_rate=0.1)
    # Define the optimizer
    optimizer = OPTIMIZER(model.parameters(), lr=1e-4, weight_decay=1.6e-4)
    trainer = Trainer(MAX_EPOCHS, FREQ_INFO, FREQ_SAVE)
    
    if TRAIN:
        trainer.fit(model, train_dataloader, valid_dataloader, optimizer)
        torch.save(model.state_dict(), 'model.pth')
    else:
        model.load_state_dict(torch.load('model.pth'))
    
    # losses, sdc_scores = trainer.validate(model, valid_dataloader)
    # losses = [torch.mean(loss).item() for loss in losses]
    # print(f'average_loss: {np.mean(losses)}')
    
    random_img = next(iter(valid_dataloader))[0][0]
    random_img = random_img.unsqueeze(0).to(DEVICE)
    prediction = model(random_img)
    print(prediction.shape)
    
    # Save the image
    org_img = transforms.ToPILImage()(random_img.squeeze(0).cpu().detach())
    pred_img = transforms.ToPILImage()(prediction.squeeze(0).cpu().detach())
    org_img.save('original.png')
    pred_img.save('prediction.png')