from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torch import Tensor


IMAGES_PATH = Path() / "images"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="eps", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
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
        freq_save = self.freq_save
        freq_info = self.freq_info

        for epoch in range(1, self.max_epochs + 1):
            # train step
            loss = None
            for frames, masks in train_dataloader:
                loss = self.train_step(model, frames, masks, optimizer)

            if epoch % freq_info == 0:
                print(f'Epoch {epoch}: loss = {loss: .5f}')

            if epoch % freq_save == 0:
                losses_all, dsc_scores_all = [], []
                for frames_val, masks_val in valid_dataloader:
                    losses, dsc_scores = self.val_step(model, frames_val, masks_val)
                    losses_all.append([losses])
                    dsc_scores_all.append([dsc_scores])
                print(f'Epoch {epoch}: val-loss={torch.reduce_mean(tf.concat(losses_all, axis=0)): .5f}, val-DSC={tf.reduce_mean(tf.concat(dsc_scores_all, axis=0)): .5f}')
                tf.saved_model.save(model, os.path.join(save_path, 'epoch{:d}'.format(epoch)))
                tf.print('Model saved.')
