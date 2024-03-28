import torch

from pathlib import Path
from os import cpu_count
from torchvision.transforms.v2 import Compose, RandomResizedCrop, ToImage, ToDtype, Resize
from models.mae import mae_vit_base_patch16_dec512d8b
from torch.optim import AdamW
from models.nn import PetModel
from data.augmentation import CannyEdgeDetection, MaskPreprocessing
from models.resunet import ResUNet


# General
IMAGES_PATH = Path("images")
MODEL_CHECKPOINTS_PATH = Path("models/checkpoints")
SEED = 42
N_CPU = cpu_count()

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_built():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
print(f"Using device: {DEVICE}")

# Data
DATA_ROOT = Path("data")

# Pre-training
PRE_TRAINING_TRANSFORMS = Compose([
    RandomResizedCrop(224),
    ToImage(),
    ToDtype(torch.float32, scale=True)
])
PRE_TRAINING_BATCH_SIZE = 16
PRE_TRAINING_MODEL = mae_vit_base_patch16_dec512d8b
PRE_TRAINING_OPTIMIZER = AdamW
PRE_TRAINING_MAX_EPOCHS = 1
PRE_TRAINING_FREQ_INFO = 1
PRE_TRAINING_FREQ_SAVE = 100
MASK_RATIO = .75

# Fine-tuning
FINE_TUNING_TRANSFORMS = Compose([
    CannyEdgeDetection(100, 200),
    MaskPreprocessing(),
    Resize((224, 224)),
    ToImage(),
    ToDtype(torch.float32, scale=True)
])
FINE_TUNING_BATCH_SIZE = 16
FINE_TUNING_MODEL = ResUNet
FINE_TUNING_OPTIMIZER = AdamW
FINE_TUNING_MAX_EPOCHS = 1
FINE_TUNING_FREQ_INFO = 1
FINE_TUNING_FREQ_SAVE = 100

# Testing
TESTING_BATCH_SIZE = 16
