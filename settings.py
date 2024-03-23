from pathlib import Path
from os import cpu_count
from models.nn import PetModel
from models.resunet import ResUNet
from torch.optim import AdamW


# General
IMAGES_PATH = Path("images")
MODEL_CHECKPOINTS_PATH = Path("models/checkpoints")
SEED = 42
N_CPU = cpu_count()

# Data
OxfordIIITPet_DATA_ROOT = Path("data/oxford-iiit-pet")

# Model
# MODEL = PetModel
MODEL = ResUNet

# Fine-tuning
FINE_TUNING_BATCH_SIZE = 16
FINE_TUNING_OPTIMIZER = AdamW
FINE_TUNING_MAX_EPOCHS = 1
FINE_TUNING_FREQ_INFO = 1
FINE_TUNING_FREQ_SAVE = 100

# Testing
TESTING_BATCH_SIZE = 16
