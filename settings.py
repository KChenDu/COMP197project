import torch

from pathlib import Path
from os import cpu_count
from torchvision.transforms.v2 import Compose, RandomResizedCrop, ToImage, ToDtype, Resize
from torch import float32
from models.mae import mae_vit_base_patch16_dec512d8b
from functools import partial
from torch.optim import AdamW
from models.nn import PetModel
from data.augmentation import CannyEdgeDetection, MaskPreprocessing
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype
from models.smp_unet import SMPMiTUNet

# General
IMAGES_PATH = Path("images")
MODEL_CHECKPOINTS_PATH = Path("models/checkpoints")
SEED = 42
DEVICE_COUNT = cpu_count()

if torch.cuda.is_available():
    DEVICE_COUNT = torch.cuda.device_count()
    DEVICE = "cuda"
    print(f"[Using CUDA] Found {DEVICE_COUNT} GPU(s) available.")
elif torch.backends.mps.is_built():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
    print(f"[Using CPU] Found {DEVICE_COUNT} CPU(s) available.")

# DEVICE = torch.device(DEVICE)
torch.set_default_device(DEVICE)

def SETUP_DEVICE():
    if DEVICE == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.multiprocessing.set_start_method('spawn')
    pass

# Data
DATA_ROOT = Path("data")

# Pre-training
PRE_TRAINING_TRANSFORM = Compose([
    RandomResizedCrop(224),
    ToImage(),
    ToDtype(float32, scale=True)
])
PRE_TRAINING_BATCH_SIZE = 16
PRE_TRAINING_MODEL = mae_vit_base_patch16_dec512d8b
PRE_TRAINING_OPTIMIZER = partial(AdamW, lr=1.5e-4 * PRE_TRAINING_BATCH_SIZE / 256., weight_decay=.05, betas=(.9, .95))
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
    ToDtype(float32, scale=True)
])
FINE_TUNING_BATCH_SIZE = 16
FINE_TUNING_MODEL = SMPMiTUNet
FINE_TUNING_OPTIMIZER = AdamW
FINE_TUNING_MAX_EPOCHS = 1
FINE_TUNING_FREQ_INFO = 1
FINE_TUNING_FREQ_SAVE = 100

# Testing
TESTING_BATCH_SIZE = 16
