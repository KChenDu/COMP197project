import torch
from settings import (SEED, 
                      DEVICE,
                      DATA_ROOT, 
                      TESTING_BATCH_SIZE as BATCH_SIZE, 
                      FINE_TUNING_TRANSFORMS as TRANSFORMS, 
                      FINE_TUNING_MODEL as MODEL, 
                      FINE_TUNED_MODEL)
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from os import cpu_count
from utils import Tester



if __name__ == '__main__':
    torch.manual_seed(SEED)

    # Device setup
    if DEVICE == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.multiprocessing.set_start_method('spawn')
        num_workers = 0
    elif DEVICE in ['mps', 'cpu']:
        num_workers = cpu_count()
    else:
        raise ValueError
    
    test_dataset = OxfordIIITPet(DATA_ROOT, split='test', target_types='segmentation', transforms=TRANSFORMS, download=True)

    # Create the dataloaders
    test_dataloader = DataLoader(test_dataset, BATCH_SIZE, num_workers=num_workers)

    # Import the model
    checkpoint = torch.load(FINE_TUNED_MODEL) #  model_pre_trained_ImageNet_20.pth
    encoder_state_dict = checkpoint['model_state_dict']
    model = MODEL()
    model.load_state_dict(encoder_state_dict, strict=False)

    tester = Tester()
    avg_loss, avg_accuracy = tester.test(model, test_dataloader)
