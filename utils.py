import torch
import math
from torch.utils.data import DataLoader
from settings import IMAGES_PATH, DEVICE
from matplotlib import pyplot as plt
from torch import Tensor, no_grad, mean
from abc import ABC, abstractmethod
from metrics import dice_loss, dice_binary, dice_score, segment_accuracy
from tqdm import tqdm
from loguru import logger
from datetime import datetime
from settings import MODEL_CHECKPOINTS_PATH
from torchvision.utils import save_image


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    path = IMAGES_PATH / (f"{fig_id}." + fig_extension)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension)

def pre_process(images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    # TODO: This part is not a good practice, we'll try to do the dtype conversion in the dataloader
    images = torch.stack([torch.tensor(image) for image in images]).float()
    labels = torch.stack([torch.tensor(label) for label in labels]).float()
    return images, labels

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args['warmup_epochs']:
        lr = args['lr'] * epoch / args['warmup_epochs']
    else:
        lr = args['min_lr'] + (args['lr'] - args['min_lr']) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args['warmup_epochs']) / (args['epochs'] - args['warmup_epochs'])))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

class BaseTrainer(ABC):
    def __init__(self, max_epochs: int = 1, freq_info: int = 1, freq_save: int = 100, device: str = DEVICE):
        self.max_epochs = max_epochs
        self.freq_info = freq_info
        self.freq_save = freq_save
        logger.info("device: " + device)
        self.device = torch.device(device)

    def plot_accuracy(self, accuracies: list[float]) -> None:
        plt.figure()
        plt.plot(accuracies, marker='o', linestyle='-', color='b')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epoch')
        save_fig('accuracy_vs_epoch')
        plt.show()
        
    def plot_loss(self, losses: list[float]) -> None:
        plt.figure()
        plt.plot(losses, marker='o', linestyle='-', color='r')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        save_fig('loss_vs_epoch')
        plt.show()
        
    def plot_sdc_score(self, sdc_scores: list[float]) -> None:
        plt.figure()
        plt.plot(sdc_scores, marker='o', linestyle='-', color='g')
        plt.xlabel('Epoch')
        plt.ylabel('SDC Score')
        plt.title('SDC Score vs Epoch')
        save_fig('sdc_score_vs_epoch')
        plt.show()
        
    def draw_predictions(self, model: torch.nn.Module, valid_dataloader: DataLoader, print_info=False, save_img=True, max_samples = 16, tag=''):
        
            frames, masks = next(iter(valid_dataloader))
            
            num_samples = frames.size(0)
            
            if num_samples > max_samples:
                num_samples = max_samples
                frames = frames[:num_samples]
                masks = masks[:num_samples]
                predicts = predicts[:num_samples]
            
            frames, masks = frames.to(self.device), masks.to(self.device)
            frames, masks = pre_process(frames, masks)
            predicts = model(frames)
            
            extra_info = {}
            if print_info:
                extra_info['loss'] = dice_loss(predicts, masks)
                extra_info['accuracy'] = segment_accuracy(predicts, masks)
                extra_info['dice_score'] = dice_score(predicts, masks)
            
            fig, axs = plt.subplots(num_samples, 4, figsize=(12, 3*num_samples))
            for j in range(num_samples):
                axs[j, 0].imshow(frames[j].cpu().numpy().transpose(1, 2, 0))
                axs[j, 0].set_title('Image')
                axs[j, 1].imshow(masks[j].cpu().numpy().transpose(1, 2, 0))
                axs[j, 1].set_title('Label Mask')
                axs[j, 2].imshow(predicts[j].cpu().detach().numpy().transpose(1, 2, 0))
                axs[j, 2].set_title('Predicted Mask')
                # axs[j, 3].imshow(np.ones(predicts.shape[2:])*255, cmap='gray')
                for ax in axs[j]:
                    ax.axis('off')
                if print_info:
                    info = '\n'.join([f'{k}: {v[j]: .5f}' for k, v in extra_info.items()])
                    axs[j, 3].text(0.1, 0.5, info, fontsize=12, color='black')
            fig.suptitle(f'Sample {tag}')
            if save_img:
                save_fig(f'prediction_{tag}')
            plt.show()

    @staticmethod
    @abstractmethod
    def training_step(*args) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def fit(self, *args):
        raise NotImplementedError


class PreTrainer(BaseTrainer):
    def __init__(self, max_epochs: int = 1, freq_info: int = 1, freq_save: int = 100, device: str = DEVICE):
        super().__init__(max_epochs, freq_info, freq_save, device)
        self.lr = 0.001
        self.min_lr = 0.0001
        self.warmup_epochs = 40

    @staticmethod
    def training_step(model, images: Tensor, optimizer, mask_ratio: float) -> Tensor:
        loss, _, _ = model(images, mask_ratio)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def fit(self, model, train_dataloader, optimizer, mask_ratio: float):

        # LR scheduler
        args = {
            'lr': self.lr,
            'min_lr': self.min_lr,
            'warmup_epochs': self.warmup_epochs,
            'epochs': self.max_epochs
        }
               
        name = type(model).__name__
        training_step = self.training_step
        freq_save = self.freq_save
        freq_info = self.freq_info
        timestamp = None
        device = self.device
        model.to(device)

        current_iteration = 0

        for epoch in range(1, self.max_epochs + 1):
            loss = None
            for frames, _ in tqdm(train_dataloader, f'Epoch {epoch}', leave=False, unit='batches'):
                # Adjust learning rate per iteration
                adjust_learning_rate(optimizer, current_iteration / len(train_dataloader) + epoch, args)
                current_iteration += 1
                loss = training_step(model, frames.to(device), optimizer, mask_ratio)

            if epoch % freq_info < 1:
                logger.info(f'Epoch {epoch}: loss = {loss: .5f}')

            if epoch % freq_save < 1:
                if timestamp is None:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                
                save_dir = MODEL_CHECKPOINTS_PATH / name / timestamp
                save_dir.mkdir(parents=True, exist_ok=True)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, save_dir / f'epoch_{epoch: d}')
                logger.info('Model saved.')

class FineTuner(BaseTrainer):
    def __init__(self, max_epochs: int = 1, freq_info: int = 1, freq_save: int = 100, device: str = DEVICE):
        super().__init__(max_epochs, freq_info, freq_save, device)

    @staticmethod
    def training_step(model, images: Tensor, labels: Tensor, optimizer) -> Tensor:
        images, labels = pre_process(images, labels)
        loss = mean(dice_loss(model(images), labels))
        acc = mean(segment_accuracy(model(images), labels))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, acc

    @staticmethod
    def validation_step(model, images: Tensor, labels: Tensor) -> Tensor:
        images, labels = pre_process(images, labels)
        predicts = model(images)
        loss = dice_binary(predicts, labels)
        acc = segment_accuracy(predicts, labels)
        return loss, acc

    @no_grad()
    def validate(self, model, valid_dataloader) -> list:
        model.eval()
        device = self.device
        validation_step = self.validation_step

        losses_all = [None] * len(valid_dataloader)
        acc_all = [None] * len(valid_dataloader)
        for i, (frames, masks) in enumerate(valid_dataloader):
            losses_all[i], acc_all[i] = validation_step(model, frames.to(device), masks.to(device))
        model.train()

        return losses_all, acc_all

    def fit(self, model, train_dataloader, valid_dataloader, optimizer):
        name = type(model).__name__
        training_step = self.training_step
        validate = self.validate
        freq_save = self.freq_save
        freq_info = self.freq_info
        timestamp = None
        device = self.device
        model.to(device)

        for epoch in range(1, self.max_epochs + 1):
            loss = None
            acc = None
            for frames, masks in tqdm(train_dataloader, f'epoch {epoch}', leave=False, unit='batches'):
                loss, acc = training_step(model, frames.to(device), masks.to(device), optimizer)

            if epoch % freq_info < 1:
                logger.info(f'Epoch {epoch}: loss = {loss: .5f}, accuracy = {acc: .5f}')

            self.draw_predictions(model, valid_dataloader, print_info=True, save_img=True, tag=epoch)

            if epoch % freq_save < 1:
                losses_all, acc_all = validate(model, valid_dataloader)
                val_loss = mean(torch.tensor(losses_all))
                val_mean = mean(torch.tensor(acc_all))
                logger.info(f'Epoch {epoch}: val-loss = {val_loss: .5f}, val-accuracy = {val_mean: .5f}')
                
                if timestamp is None:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, MODEL_CHECKPOINTS_PATH / name / timestamp / f'epoch_{epoch: d}')
                logger.info('Model saved.')


        
class Tester:
    def __init__(self, device: str = DEVICE):
        self.device = torch.device(device)
        
    @no_grad()
    def test(self, model, test_dataloader) -> tuple[float, float]:
        model.eval()  
        device = self.device
        total_loss, total_accuracy = [], []

        for frames_test, masks_test in tqdm(test_dataloader, desc='Testing', unit='batches'):
            
            frames_test, masks_test = pre_process(frames_test, masks_test)
            frames_test, masks_test = frames_test.to(device), masks_test.to(device)

            predicts_test = model(frames_test)
            total_loss += [mean(dice_loss(predicts_test, masks_test))]
            total_accuracy += [mean(dice_binary(predicts_test, masks_test))]

        avg_loss = mean(torch.tensor(total_loss))
        avg_accuracy = mean(torch.tensor(total_accuracy))

        model.train()  
        return avg_loss, avg_accuracy

        
