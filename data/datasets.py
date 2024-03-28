from torch.utils.data import Dataset
from pathlib import Path
from torch import is_tensor
from PIL.Image import open
from segmentation_models_pytorch.datasets import OxfordPetDataset
from PIL.Image import fromarray


class KaggleCatsAndDogsDataset(Dataset):
    """Kaggle Cats and Dogs Dataset"""
    def __init__(self, root_dir: str, transform=None):
        """
        Arguments:
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.image_paths = list(Path(root_dir).glob('PetImages/**/*.jpg'))  # Assuming images are stored as .jpg

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = open(img_path).convert('RGB')  # Ensure image is in RGB format

        if self.transform:
            image = self.transform(image)

        return image, -1


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):
        sample = super().__getitem__(*args, **kwargs)
        return fromarray(sample["image"]), fromarray(sample["mask"])
