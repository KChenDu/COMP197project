from os import cpu_count
from pathlib import Path
from models.nn import PetModel
from torch.optim import AdamW


# General
SEED = 42
N_CPU = cpu_count()

# Data
DATA_ROOT = Path('data')
OxfordIIITPet_DATA_ROOT = DATA_ROOT / 'oxford-iiit-pet'

# Model
MODEL = PetModel

# Fine-tuning
FINE_TUNING_BATCH_SIZE = 16
FINE_TUNING_OPTIMIZER = AdamW
FINE_TUNING_MAX_EPOCHS = 1
FINE_TUNING_FREQ_INFO = 1
FINE_TUNING_FREQ_SAVE = 100

# Testing
TESTING_BATCH_SIZE = 16
