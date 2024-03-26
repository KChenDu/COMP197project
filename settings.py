import torch

from pathlib import Path
from os import cpu_count
from models.nn import PetModel
from models.resunet import ResUNet
from data.augmentation import CannyEdgeDetection, MaskPreprocessing
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype


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

# Data
DATA_ROOT = Path("data")

# Model
# MODEL = PetModel
MODEL = ResUNet

# Fine-tuning
FINE_TUNING_TRANSFORMS = Compose([
    CannyEdgeDetection(100, 200),
    MaskPreprocessing(),
    Resize((224, 224)),
    ToImage(),
    ToDtype(torch.float32, scale=True)
])
FINE_TUNING_BATCH_SIZE = 16
FINE_TUNING_OPTIMIZER = torch.optim.AdamW
FINE_TUNING_MAX_EPOCHS = 1
FINE_TUNING_FREQ_INFO = 1
FINE_TUNING_FREQ_SAVE = 100

# Testing
TESTING_BATCH_SIZE = 16
