import logging
import torch

from settings import IMAGES_PATH, DEVICE, MODEL_CHECKPOINTS_PATH
from matplotlib import pyplot as plt
from torchvision.transforms.v2 import Normalize
from torch import Tensor, mean, no_grad, tensor
from abc import ABC, abstractmethod
from torch.nn import Module
from torch.optim import Optimizer
from datetime import datetime
from torch.utils.data import DataLoader
from models.util.lr_sched import adjust_learning_rate
from tqdm import tqdm
from metrics import dice_loss, dice_binary, binary_accuracy


def setup_logger(file_handler_path):
    LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_handler_path)
    
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    path = IMAGES_PATH / (f"{fig_id}." + fig_extension)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension)


def pre_process(images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    return Normalize((.485, .456, .406), (.229, .224, .225))(images), labels


class BaseTrainer(ABC):
    def __init__(self, max_epochs: int = 1, freq_info: int = 1, freq_save: int = 100, device: str = DEVICE):
        self.max_epochs = max_epochs
        self.freq_info = freq_info
        self.freq_save = freq_save
        self.device = torch.device(device)
        self.timestamp = None
        self.logger = None

    def save_model(self, model: Module, epoch: int, optimizer: Optimizer,
                   loss: Tensor, accuracy: Tensor = None, losses: list[Tensor] = None, accuracies: list[Tensor] = None,
                   val_loss: Tensor = None, val_accuracy: Tensor = None, val_losses: list[Tensor] = None, val_accuracies: list[Tensor] = None) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = MODEL_CHECKPOINTS_PATH / type(model).__name__ / self.timestamp
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'losses': losses,
            'accuracies': accuracies,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }, save_dir / f'epoch_{epoch:d}.pt')
        self.logger.info('Model saved.')

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
        logger = setup_logger('log/pre-training.log')
        logger.info("device: " + device)
        self.logger = logger

    @staticmethod
    def training_step(model: Module, images: Tensor, optimizer: Optimizer, mask_ratio: float) -> Tensor:
        loss, _, _ = model(images, mask_ratio)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def fit(self, model: Module, train_dataloader: DataLoader, optimizer: Optimizer, mask_ratio: float, args: dict) -> None:
        save_model = self.save_model
        freq_save = self.freq_save
        freq_info = self.freq_info
        logger = self.logger
        training_step = self.training_step
        max_epochs = self.max_epochs
        device = self.device

        model.to(device)
        losses = []
        loss = None
        length = len(train_dataloader)

        for epoch in range(1, max_epochs + 1):
            for data_iter_step, (frames, _) in enumerate(tqdm(train_dataloader, f'Epoch {epoch}', leave=False, unit='batches')):
                # we use a per iteration (instead of per epoch) lr scheduler
                adjust_learning_rate(optimizer, data_iter_step / length + epoch, args)
                loss = training_step(model, frames.to(device), optimizer, mask_ratio)

            if epoch % freq_info < 1:
                logger.info(f'Epoch {epoch}: loss = {loss: .5f}')

            losses.append(loss)

            if epoch % freq_save < 1:
                save_model(model, epoch, optimizer, loss)

        if max_epochs % freq_save > 0:
            save_model(model, max_epochs, optimizer, loss, losses)


class FineTuner(BaseTrainer):
    def __init__(self, max_epochs: int = 1, freq_info: int = 1, freq_save: int = 100, device: str = DEVICE):
        super().__init__(max_epochs, freq_info, freq_save, device)
        logger = setup_logger('log/fine-tuning.log')
        logger.info("device: " + device)
        self.logger = logger

    def plot_accuracy(self, accuracies: list[float]) -> None:
        accuracies = [accu.item() for accu in accuracies]
        plt.figure()
        plt.plot(accuracies, marker='o', linestyle='-', color='b')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epoch')
        save_fig('accuracy_vs_epoch')
        plt.show()

    def plot_loss(self, losses: list[float]) -> None:
        losses = [loss.item() for loss in losses]
        plt.figure()
        plt.plot(losses, marker='o', linestyle='-', color='r')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        save_fig('loss_vs_epoch')
        plt.show()

    def plot_sdc_score(self, sdc_scores: list[float]) -> None:
        sdc_scores = [score.item() for score in sdc_scores]
        plt.figure()
        plt.plot(sdc_scores, marker='o', linestyle='-', color='g')
        plt.xlabel('Epoch')
        plt.ylabel('SDC Score')
        plt.title('SDC Score vs Epoch')
        save_fig('sdc_score_vs_epoch')
        plt.show()

    def draw_predictions(self, model: Module, valid_dataloader: DataLoader, print_info=False, save_img=True,
                         max_samples=16, tag=''):

        frames, masks = next(iter(valid_dataloader))

        num_samples = frames.size(0)

        if num_samples > max_samples:
            num_samples = max_samples
            frames = frames[:num_samples]
            masks = masks[:num_samples]
            # predicts = predicts[:num_samples]

        frames, masks = frames.to(self.device), masks.to(self.device)
        frames, masks = pre_process(frames, masks)
        predicts = model(frames)
        frames = frames.type(torch.uint8)

        extra_info = {}
        if print_info:
            extra_info['loss'] = dice_loss(predicts, masks)
            extra_info['accuracy'] = binary_accuracy(predicts, masks)
            extra_info['bin_dice_score'] = dice_binary(predicts, masks)

        fig, axs = plt.subplots(num_samples, 4, figsize=(12, 3 * num_samples))
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
    def training_step(model: Module, images: Tensor, labels: Tensor, optimizer: Optimizer) -> tuple[Tensor, Tensor]:
        images, labels = pre_process(images, labels)
        predictions = model(images)
        accuracy = binary_accuracy(predictions, labels)
        loss = mean(dice_loss(predictions, labels))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, accuracy

    @staticmethod
    def validation_step(model: Module, images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        images, labels = pre_process(images, labels)
        predicts = model(images)
        accuracy = binary_accuracy(predicts, labels)
        loss = mean(dice_loss(predicts, labels))
        return loss, accuracy

    @no_grad()
    def validate(self, model: Module, valid_dataloader: DataLoader) -> tuple[list[Tensor], list[Tensor]]:
        validation_step = self.validation_step
        device = self.device

        length = len(valid_dataloader)
        losses_all = [None] * length
        accuracies_all = [None] * length

        model.eval()
        for i, (frames, masks) in enumerate(valid_dataloader):
            losses_all[i], accuracies_all[i] = validation_step(model, frames.to(device), masks.to(device))
        model.train()

        return losses_all, accuracies_all

    def fit(self, model: Module, train_dataloader: DataLoader, valid_dataloader: DataLoader, optimizer: Optimizer) -> None:
        save_model = self.save_model
        freq_save = self.freq_save
        freq_info = self.freq_info
        logger = self.logger
        training_step = self.training_step
        validate = self.validate
        max_epochs = self.max_epochs
        device = self.device

        model.to(device)
        val_losses = []
        val_loss = None
        val_accuracies = []
        val_accuracy = None
        losses = []
        loss = None
        accuracies = []
        accuracy = None

        for epoch in range(1, max_epochs + 1):
            for frames, masks in tqdm(train_dataloader, f'epoch {epoch}', leave=False, unit='batches'):
                loss, accuracy = training_step(model, frames.to(device), masks.to(device), optimizer)

            val_losses_all, val_accuracies_all = validate(model, valid_dataloader)
            val_loss = mean(tensor(val_losses_all))
            val_accuracy = mean(tensor(val_accuracies_all))

            if epoch % freq_info < 1:
                logger.info(f'Epoch {epoch}: loss = {loss: .5f}, accuracy = {accuracy: .5f}, val_loss = {val_loss: .5f}, val_accuracy = {val_accuracy: .5f}')

            losses.append(loss)
            accuracies.append(accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            # self.draw_predictions(model, valid_dataloader, print_info=True, save_img=True, tag=epoch)

            if epoch % freq_save < 1:
                save_model(model, epoch, optimizer, loss, accuracy)

        if max_epochs % freq_save > 0:
            save_model(model, max_epochs, optimizer, loss, accuracy, losses, accuracies, val_loss, val_accuracy, val_losses, val_accuracies)
        # self.plot_loss(losses)
        # self.plot_accuracy(self.accuracies)
        # self.plot_sdc_score(self.sdc_scores)

        
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
