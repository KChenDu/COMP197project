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
from loguru import logger


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
    
    torch.set_default_device(DEVICE)
    
    test_dataset = OxfordIIITPet(DATA_ROOT, split='test', target_types='segmentation', transforms=TRANSFORMS, download=True)

    # Create the dataloaders
    test_dataloader = DataLoader(test_dataset, BATCH_SIZE, num_workers=num_workers)

    # Import the model
    checkpoint = torch.load(FINE_TUNED_MODEL) #  model_pre_trained_ImageNet_20.pth
    encoder_state_dict = checkpoint['model_state_dict']
    model = MODEL()
    model.load_state_dict(encoder_state_dict, strict=False)

    tester = Tester()
    # avg_loss, avg_accuracy = tester.test(model, test_dataloader)
    # logger.info(f'For testing: val-- loss = {avg_loss: .5f}, val-- DSC = {avg_accuracy: .5f}')
    
    # Draw predictions for 5 models
    BASE_PATH = 'models/checkpoints/02-fine-tuned/'
    model_paths = [
        'Kaggle_100oxPet/epoch_60.pt',
        'ImageNet_100oxPet/epoch_60.pt',
        'ImageNet_100oxPet_withEdge/epoch_60.pt',
        'ImageNet_100oxPet_withBlur/epoch_60.pt',
        #'ImageNet_100oxPet_withBlurAndEdge/epoch_60.pt'
    ]
    states = [(BASE_PATH + path) for path in model_paths]
    tester.draw_predictions_for_models(model = model,
                                       saved_states=states,
                                       tags = ['Kaggle', 'ImageNet', 'ImageNet with Edge', 'ImageNet with Blur'], # , 'ImageNet with Edge and Blur'],
                                       test_dataloader = test_dataloader,
                                       save_img_file='test_predictions')
    
    