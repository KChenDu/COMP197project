import torch

from settings import IMAGES_PATH
from matplotlib import pyplot as plt
from torch import Tensor, mean, concat
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
    raise NotImplementedError


def dice_loss(ps: Tensor, ts: Tensor) -> Tensor:
    raise NotImplementedError


def dice_binary(ps: Tensor, ts: Tensor) -> Tensor:
    raise NotImplementedError


class Trainer:
    def __init__(self, max_epochs: int = 1, freq_info: int = 1, freq_save: int = 100):
        self.max_epochs = max_epochs
        self.freq_info = freq_info
        self.freq_save = freq_save

    @staticmethod
    def train_step(model, images: Tensor, labels: Tensor, optimizer) -> Tensor:
        images, labels = pre_process(images, labels)
        # Q: add data augmentation
        loss = dice_loss(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    @staticmethod
    def val_step(seg_net, images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        images, labels = pre_process(images, labels)
        predicts = seg_net(images, training=False)
        with torch.no_grad():
            losses = dice_loss(predicts, labels)
            dsc_scores = dice_binary(predicts, labels)
        return losses, dsc_scores

    def fit(self, model, train_dataloader, valid_dataloader, optimizer):
        name = type(model).__name__
        train_step = self.train_step
        val_step = self.val_step
        freq_save = self.freq_save
        freq_info = self.freq_info
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        for epoch in range(1, self.max_epochs + 1):
            loss = None
            for frames, masks in train_dataloader:
                loss = train_step(model, frames, masks, optimizer)

            if epoch % freq_info < 1:
                logger.info(f'Epoch {epoch}: loss = {loss: .5f}')

            if epoch % freq_save < 1:
                losses_all, dsc_scores_all = [], []
                for frames, masks in valid_dataloader:
                    losses, dsc_scores = val_step(model, frames, masks)
                    losses_all.append(losses)
                    dsc_scores_all.append(dsc_scores)
                # Dangerous: this piece below is not sure for me and not tested, please (everyone who looking) help check it comparing with tutorial's code
                logger.info(f'Epoch {epoch}: val-loss = {mean(concat(losses_all)): .5f}, val-DSC = {mean(concat(dsc_scores_all)): .5f}')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, MODEL_CHECKPOINTS_PATH / name / timestamp / f'epoch_{epoch: d}')
                logger.info('Model saved.')

    @staticmethod
    def validate(model, valid_dataloader):
        raise NotImplementedError


class Tester:
    def __init__(self):
        # initialize parameters here as needed
        pass

    @staticmethod
    def test(model, test_dataloader) -> float:
        return 1.
