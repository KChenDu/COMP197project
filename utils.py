import torch

from settings import IMAGES_PATH, DEVICE
from matplotlib import pyplot as plt
from torch import Tensor, no_grad, mean
from abc import ABC, abstractmethod
from metrics import dice_loss, dice_binary
from tqdm import tqdm
from loguru import logger
from datetime import datetime
from settings import MODEL_CHECKPOINTS_PATH


def save_fig(fig_id, tight_layout=True, fig_extension="eps", resolution=300):
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    path = IMAGES_PATH / (f"{fig_id}." + fig_extension)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def pre_process(images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    # TODO: This part is not a good practice, we'll try to do the dtype conversion in the dataloader
    images = torch.stack([image.clone().detach() for image in images]).float()
    labels = torch.stack([label.clone().detach() for label in labels]).float()
    return images, labels


class BaseTrainer(ABC):
    def __init__(self, max_epochs: int = 1, freq_info: int = 1, freq_save: int = 100, device: str = DEVICE):
        self.max_epochs = max_epochs
        self.freq_info = freq_info
        self.freq_save = freq_save
        self.device = torch.device(device)

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

    @staticmethod
    def training_step(model, images: Tensor, optimizer, mask_ratio: float) -> Tensor:
        loss, _, _ = model(images, mask_ratio)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def fit(self, model, train_dataloader, optimizer, mask_ratio: float):
        name = type(model).__name__
        training_step = self.training_step
        freq_save = self.freq_save
        freq_info = self.freq_info
        timestamp = None
        device = self.device
        model.to(device)

        for epoch in range(1, self.max_epochs + 1):
            loss = None
            for frames, masks in tqdm(train_dataloader, f'epoch {epoch}', leave=False, unit='batches'):
                loss = training_step(model, frames.to(device), optimizer, mask_ratio)

            if epoch % freq_info < 1:
                logger.info(f'Epoch {epoch}: loss = {loss: .5f}')

            if epoch % freq_save < 1:
                if timestamp is None:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, MODEL_CHECKPOINTS_PATH / name / timestamp / f'epoch_{epoch: d}')
                logger.info('Model saved.')


class FineTuner(BaseTrainer):
    def __init__(self, max_epochs: int = 1, freq_info: int = 1, freq_save: int = 100, device: str = DEVICE):
        super().__init__(max_epochs, freq_info, freq_save, device)

    @staticmethod
    def training_step(model, images: Tensor, labels: Tensor, optimizer) -> Tensor:
        images, labels = pre_process(images, labels)
        loss = mean(dice_loss(model(images), labels))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    @staticmethod
    def validation_step(model, images: Tensor, labels: Tensor) -> Tensor:
        images, labels = pre_process(images, labels)
        predicts = model(images)
        loss = dice_binary(predicts, labels)
        return loss

    @no_grad()
    def validate(self, model, valid_dataloader) -> list:
        model.eval()
        device = self.device
        validation_step = self.validation_step

        losses_all = [None] * len(valid_dataloader)
        for i, (frames, masks) in enumerate(valid_dataloader):
            losses_all[i] = validation_step(model, frames.to(device), masks.to(device))
        model.train()

        return losses_all

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
            for frames, masks in tqdm(train_dataloader, f'epoch {epoch}', leave=False, unit='batches'):
                loss = training_step(model, frames.to(device), masks.to(device), optimizer)

            if epoch % freq_info < 1:
                logger.info(f'Epoch {epoch}: loss = {loss: .5f}')

            if epoch % freq_save < 1:
                losses_all = validate(model, valid_dataloader)
                logger.info(f'Epoch {epoch}: val-loss = {mean(torch.tensor(losses_all)): .5f}')
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
    def __init__(self):
        # TODO: add configurable parameters for tester
        pass

    @staticmethod
    @no_grad()
    def test(model, test_dataloader) -> float:
        # TODO: implement this (might be similar to Trainer.validate)
        return 1.
