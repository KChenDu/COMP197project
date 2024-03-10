import torch

from settings import IMAGES_PATH
from matplotlib import pyplot as plt
from torch import Tensor, no_grad, mean, concat
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
    # TODO: implement this (depends on needs, can be not necessary)
    raise NotImplementedError


def dice_loss(ps: Tensor, ts: Tensor) -> Tensor:
    # TODO: implement this (but there is a chance that it is already implemented in segmentation_models_pytorch, check it first)
    raise NotImplementedError


def dice_binary(ps: Tensor, ts: Tensor) -> Tensor:
    # TODO: implement this (but there is a chance that it is already implemented in segmentation_models_pytorch, check it first)
    raise NotImplementedError


class Trainer:
    def __init__(self, max_epochs: int = 1, freq_info: int = 1, freq_save: int = 100):
        self.max_epochs = max_epochs
        self.freq_info = freq_info
        self.freq_save = freq_save

    @staticmethod
    def training_step(model, images: Tensor, labels: Tensor, optimizer) -> Tensor:
        images, labels = pre_process(images, labels)
        # TODO: add data augmentation (maybe here or not here)
        loss = dice_loss(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    @staticmethod
    def validation_step(seg_net, images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        images, labels = pre_process(images, labels)
        predicts = seg_net(images)
        losses = dice_loss(predicts, labels)
        dsc_scores = dice_binary(predicts, labels)
        return losses, dsc_scores

    @no_grad()
    def validate(self, model, valid_dataloader):
        validation_step = self.validation_step

        losses_all, dsc_scores_all = [], []
        for frames, masks in valid_dataloader:
            losses, dsc_scores = validation_step(model, frames, masks)
            losses_all.append(losses)
            dsc_scores_all.append(dsc_scores)

        return losses_all, dsc_scores_all

    def fit(self, model, train_dataloader, valid_dataloader, optimizer):  # TODO: check and test it
        name = type(model).__name__
        training_step = self.training_step
        validate = self.validate
        freq_save = self.freq_save
        freq_info = self.freq_info
        timestamp = None

        for epoch in range(1, self.max_epochs + 1):
            loss = None
            for frames, masks in train_dataloader:
                loss = training_step(model, frames, masks, optimizer)

            if epoch % freq_info < 1:
                logger.info(f'Epoch {epoch}: loss = {loss: .5f}')

            if epoch % freq_save < 1:
                losses_all, dsc_scores_all = validate(model, valid_dataloader)
                # TODO: Danger! This piece below is not sure for me and not tested, please everyone who looking help check it comparing with tutorial's code
                logger.info(f'Epoch {epoch}: val-loss = {mean(concat(losses_all)): .5f}, val-DSC = {mean(concat(dsc_scores_all)): .5f}')
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
