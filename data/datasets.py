from segmentation_models_pytorch.datasets import OxfordPetDataset
from PIL.Image import fromarray


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):
        sample = super().__getitem__(*args, **kwargs)
        return fromarray(sample["image"]), fromarray(sample["mask"])
