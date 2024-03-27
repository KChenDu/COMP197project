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

# Data
OxfordIIITPet_DATA_ROOT = Path("data/oxford-iiit-pet")

# Model
# MODEL = PetModel
MODEL = ViTMaskAutoEncoder

# Fine-tuning
FINE_TUNING_BATCH_SIZE = 16
FINE_TUNING_OPTIMIZER = AdamW
FINE_TUNING_MAX_EPOCHS = 1
FINE_TUNING_FREQ_INFO = 1
FINE_TUNING_FREQ_SAVE = 100

# Testing
TESTING_BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE == torch.device("cuda"):
    print('[USING CUDA]')
    N_CPU = 1
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.fastest = True
    torch.set_default_device(DEVICE)
    torch.set_default_dtype(torch.float32)
    
    if N_CPU > 1:
        torch.multiprocessing.set_start_method("spawn")
else:
    print('[USING CPU]')
    torch.set_default_device(DEVICE)