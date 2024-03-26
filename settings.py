from pathlib import Path
from os import cpu_count
from models.nn import PetModel
from models.resunet import ResUNet
from data.augmentation import CannyEdgeDetection
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype
from torch.optim import AdamW
import torch

# General
IMAGES_PATH = Path("images")
MODEL_CHECKPOINTS_PATH = Path("models/checkpoints")
SEED = 42
N_CPU = cpu_count()

# Data
DATA_ROOT = Path("data")

# Model
# MODEL = PetModel
MODEL = ResUNet

# Fine-tuning
FINE_TUNING_TRANSFORMS = Compose([
    CannyEdgeDetection(100, 200),
    Resize((224, 224)),
    ToImage(),
    ToDtype(torch.float32, scale=True)
])
FINE_TUNING_BATCH_SIZE = 16
FINE_TUNING_OPTIMIZER = AdamW
FINE_TUNING_MAX_EPOCHS = 1
FINE_TUNING_FREQ_INFO = 1
FINE_TUNING_FREQ_SAVE = 100

# Testing
TESTING_BATCH_SIZE = 16

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if DEVICE == torch.device("cuda"):
#     print('[USING CUDA]')
#     N_CPU = 1
#
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.fastest = True
#     torch.set_default_device(DEVICE)
#     torch.set_default_dtype(torch.float32)
#
#     if N_CPU > 1:
#         torch.multiprocessing.set_start_method("spawn")
# else:
#     print('[USING CPU]')
#     torch.set_default_device(DEVICE)
