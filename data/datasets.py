import numpy as np

from segmentation_models_pytorch.datasets import OxfordPetDataset
from PIL import Image
from PIL.Image import Resampling
from torch import from_numpy


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Resampling.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Resampling.NEAREST))

        # convert to other format HWC -> CHW
        image = np.moveaxis(image, -1, 0)
        mask = np.expand_dims(mask, 0)

        return from_numpy(image), from_numpy(mask)
