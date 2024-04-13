import torch

from pathlib import Path
from torchvision.transforms import InterpolationMode
from torch import float32
from torchvision.transforms.v2 import RandomResizedCrop, RandomHorizontalFlip, ToImage, ToDtype, Normalize, Compose
from models.mae import mae_vit_pet
from torch.optim import AdamW
from functools import partial

from data.augmentation import Preprocess
from models.nn import PetModel
from data.augmentation import CannyEdgeDetection
from models.resunet import ResUNet
from models.smp_unet import SMPMiTUNet, ViTEncodedUnet


# General
IMAGES_PATH = Path("images")
MODEL_CHECKPOINTS_PATH = Path("models/checkpoints")
SEED = 42

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_built():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

# Data
DATA_ROOT = Path("data")

# Pre-training
# DATASET = 'ImageNet'
DATASET = 'KaggleCatsAndDogs'
PRE_TRAINING_TRANSFORM = Compose([
    RandomResizedCrop(224, (.2, 1.), interpolation=InterpolationMode.BICUBIC),
    RandomHorizontalFlip(),
    ToImage(),
    ToDtype(float32),
    Normalize((.485, .456, .406), (.229, .224, .225))
])
PRE_TRAINING_BATCH_SIZE = 16
PRE_TRAINING_MODEL = mae_vit_pet
LR = 1.5e-4 * PRE_TRAINING_BATCH_SIZE / 256.
PRE_TRAINING_OPTIMIZER = partial(AdamW, lr=LR, weight_decay=.05, betas=(.9, .95))
PRE_TRAINING_MAX_EPOCHS = 3
PRE_TRAINING_FREQ_INFO = 1
PRE_TRAINING_FREQ_SAVE = 2
MASK_RATIO = .75
LR_SCHED_ARGS = {
    "warmup_epochs": 40,
    "min_lr": 0.,
    "lr": 1e-3,
    "epochs": PRE_TRAINING_MAX_EPOCHS
}

# Fine-tuning
BASELINE_MODE = False  # Put in better place later
FINE_TUNING_TRANSFORMS = Compose([
    # CannyEdgeDetection(100, 200),
    Preprocess(),
])
FINE_TUNING_BATCH_SIZE = 16
# PRE_TRAINED_MODEL_FOR_FINE_TUNING = './models/checkpoints/MaskedAutoencoderViT/2024-04-05_11-09-11_ImageNet/epoch_60.pth'
FINE_TUNING_DATA_USAGE = 1 # [0.25, 0.5, 0.75, 1]
# FINE_TUNING_MODEL = SMPMiTUNet
FINE_TUNING_MODEL = ViTEncodedUnet
FINE_TUNING_OPTIMIZER = partial(AdamW, lr=1e-4, weight_decay=1.6e-4)
FINE_TUNING_MAX_EPOCHS = 1
FINE_TUNING_FREQ_INFO = 1
FINE_TUNING_FREQ_SAVE = 100

# Testing
TESTING_BATCH_SIZE = 16
FINE_TUNED_MODEL = './models/fine-tuned/epoch_60.pt'
PRE_TRAINED_MODEL = './models/pre-trained/epoch_50_Kaggle.pth'