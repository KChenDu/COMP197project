import torch

from models.smp_unet import ViTEncodedUnet
from settings import (SEED,
                      DEVICE,
                      DATA_ROOT,
                      FINE_TUNING_TRANSFORMS as TRANSFORMS,
                      FINE_TUNING_BATCH_SIZE as BATCH_SIZE,
                      PRE_TRAINED_MODEL_FOR_FINE_TUNING as PRE_TRAINED_MODEL,
                      FINE_TUNING_MODEL as MODEL,
                      FINE_TUNING_OPTIMIZER as OPTIMIZER,
                      FINE_TUNING_MAX_EPOCHS as MAX_EPOCHS,
                      FINE_TUNING_FREQ_INFO as FREQ_INFO,
                      FINE_TUNING_FREQ_SAVE as FREQ_SAVE,
                      BASELINE_MODE)
from torch import manual_seed, Generator
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import random_split, DataLoader
from utils import FineTuner
from os import cpu_count

TRAIN = True

if __name__ == '__main__':
    manual_seed(SEED)

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

    generator = Generator(DEVICE)
    # Load the train and validation datasets
    # _, train_dataset, valid_dataset = random_split(OxfordIIITPet(DATA_ROOT, target_types='segmentation', transforms=TRANSFORMS, download=True), (.8, .1, .1), generator=generator)
    train_dataset, valid_dataset = random_split(
        OxfordIIITPet(DATA_ROOT, target_types='segmentation', transforms=TRANSFORMS, download=True), (.9, .1),
        generator=generator)

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=num_workers, generator=generator)
    valid_dataloader = DataLoader(valid_dataset, BATCH_SIZE, num_workers=num_workers, generator=generator)

    # Import the model
    if MODEL is ViTEncodedUnet:
        if not BASELINE_MODE:
            # Load checkpoint
            # checkpoint = torch.load(PRE_TRAINED_MODEL)
            # model = MODEL(encoder_state_dict=checkpoint)

            checkpoint = torch.load(PRE_TRAINED_MODEL) #  model_pre_trained_ImageNet_20.pth
            encoder_state_dict = checkpoint['model_state_dict']
            model = MODEL()
            model.load_state_dict(encoder_state_dict, strict=False)
        else:
            model = MODEL(encoder_state_dict=None)
    else:
        model = MODEL()
    
    # Define the optimizer
    optimizer = OPTIMIZER(model.parameters())

    finetuner = FineTuner(MAX_EPOCHS, FREQ_INFO, FREQ_SAVE, DEVICE)

    if TRAIN:
        finetuner.fit(model, train_dataloader, valid_dataloader, optimizer)
        # Fine-tuner现在会自动保存最终模型、是否是baseline的判断就直接用timestamp吧。等找到了好的checkpoint再把它复制出来吧
        # torch.save(model.state_dict(), f'./models/model_fine_tuned_final_{"baseline" if BASELINE_MODE else "pretrained"}.pth')
    else:
        model.load_state_dict(torch.load('./models/model_fine_tuned_final.pth'))

    # losses, sdc_scores = trainer.validate(model, valid_dataloader)
    # losses = [torch.mean(loss).item() for loss in losses]
    # print(f'average_loss: {np.mean(losses)}')

    # random_img = next(iter(valid_dataloader))[0][0]
    # random_img = random_img.unsqueeze(0).to(DEVICE)
    # prediction = model(random_img)
    # print(prediction.shape)

    # # Save the image
    # org_img = transforms.ToPILImage()(random_img.squeeze(0).cpu().detach())
    # pred_img = transforms.ToPILImage()(prediction.squeeze(0).cpu().detach())
    # org_img.save('original.png')
    # pred_img.save('prediction.png')
