from settings import (SEED,
                      DATA_ROOT,
                      PRE_TRAINING_TRANSFORMS as TRANSFORMS,
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
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from utils import PreTrainer


if __name__ == '__main__':
    manual_seed(SEED)

    # Load the train and validation datasets
    train_dataset = OxfordIIITPet(DATA_ROOT, target_types='segmentation', transforms=TRANSFORMS, download=True)

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=N_CPU)

    # Import the model
    model = MODEL()
    # Define the optimizer
    optimizer = OPTIMIZER(model.parameters())

    finetuner = PreTrainer(MAX_EPOCHS, FREQ_INFO, FREQ_SAVE, DEVICE)
    finetuner.fit(model, train_dataloader, optimizer, MASK_RATIO)
