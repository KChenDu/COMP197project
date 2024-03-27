import torch

from pathlib import Path
from os import cpu_count
from models.nn import PetModel
from models.resunet import ResUNet
from data.augmentation import CannyEdgeDetection, MaskPreprocessing
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype

from models.MAEncoder import ViTMaskAutoEncoder

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

DEVICE = torch.device(DEVICE)
torch.set_default_device(DEVICE)

def SETUP_DEVICE():
    if DEVICE.type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.multiprocessing.set_start_method('spawn')
    pass

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
MODEL = ViTMaskAutoEncoder

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
