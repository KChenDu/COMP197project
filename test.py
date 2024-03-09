from settings import DATA_ROOT, OxfordIIITPet_DATA_ROOT, TESTING_BATCH_SIZE, N_CPU, MODEL
from torchvision.datasets import OxfordIIITPet
from pathlib import Path
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
from torch.utils.data import DataLoader
from utils import Tester


if __name__ == '__main__':
    # Download the dataset if it doesn't exist
    OxfordIIITPet(DATA_ROOT, "test", target_types="segmentation", download=True)
    Path.rmdir(Path("images"))

    # Load the test datasets
    test_dataset = SimpleOxfordPetDataset(OxfordIIITPet_DATA_ROOT, "test", )

    # Create the dataloader
    test_dataloader = DataLoader(test_dataset, TESTING_BATCH_SIZE, num_workers=N_CPU)

    # Import the model
    model = MODEL()

    tester = Tester()
    tester.test(model, test_dataloader)
