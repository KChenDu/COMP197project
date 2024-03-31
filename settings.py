import torch

from pathlib import Path
from os import cpu_count
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Compose, RandomResizedCrop, RandomHorizontalFlip, ToImage, Normalize, ToDtype, Resize
from torch import float32
from models.mae import mae_vit_base_patch16_dec512d8b
from functools import partial
from torch.optim import AdamW
from models.nn import PetModel
from data.augmentation import CannyEdgeDetection, MaskPreprocessing
from models.resunet import ResUNet
from models.smp_unet import SMPMiTUNet


# General
IMAGES_PATH = Path("images")
MODEL_CHECKPOINTS_PATH = Path("models/checkpoints")
SEED = 42
NUM_WORKERS = cpu_count()
PIN_MEMORY = False

if torch.cuda.is_available():
    NUM_WORKERS = 0 # turn off multi-processing because of CUDA is not very compatible with it
    # PIN_MEMORY = True # PyTorch recommends to enable pin_memory when using CUDA, but we are using custom type, so it need additional handling
    DEVICE = "cuda"
    print(f"[Using CUDA] Found {NUM_WORKERS} GPU(s) available.")
elif torch.backends.mps.is_built():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
    print(f"[Using CPU] Found {NUM_WORKERS} CPU(s) available.")

torch.set_default_device(DEVICE)


def setup_device():
    if DEVICE == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.multiprocessing.set_start_method('spawn')


# Data
DATA_ROOT = Path("data")

# Pre-training
PRE_TRAINING_TRANSFORM = Compose([
    RandomResizedCrop(224, (.2, 1.), interpolation=InterpolationMode.BICUBIC),
    RandomHorizontalFlip(),
    ToImage(),
    ToDtype(float32),
    Normalize((.485, .456, .406), (.229, .224, .225))
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
    # CannyEdgeDetection(100, 200),
    MaskPreprocessing(explicit_edge=False),
    Resize((224, 224)),
    ToImage(),
    ToDtype(float32, scale=True)
])
FINE_TUNING_BATCH_SIZE = 16
FINE_TUNING_MODEL = SMPMiTUNet
FINE_TUNING_OPTIMIZER = partial(AdamW, lr=1e-4, weight_decay=1.6e-4)
FINE_TUNING_MAX_EPOCHS = 1
FINE_TUNING_FREQ_INFO = 1
FINE_TUNING_FREQ_SAVE = 100

# Testing
TESTING_BATCH_SIZE = 16
