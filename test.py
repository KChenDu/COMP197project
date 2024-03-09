from argparse import ArgumentParser
from settings import DATA_ROOT, OxfordIIITPet_DATA_ROOT, TESTING_BATCH_SIZE as BATCH_SIZE, N_CPU, MODEL
from torchvision.datasets import OxfordIIITPet
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
from torch.utils.data import DataLoader
from torch import load
from utils import Tester


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model_checkpoint')
    args = parser.parse_args()

    # Download the dataset if it doesn't exist
    OxfordIIITPet(DATA_ROOT, "test", target_types="segmentation", download=True)

    # Load the test datasets
    test_dataset = SimpleOxfordPetDataset(OxfordIIITPet_DATA_ROOT, "test", )

    # Create the dataloader
    test_dataloader = DataLoader(test_dataset, BATCH_SIZE, num_workers=N_CPU)

    # Import the model
    model = MODEL()
    # Load the model checkpoint
    model.load_state_dict(load(args.model_checkpoint)['model_state_dict'])

    tester = Tester()
    metric = tester.test(model, test_dataloader)
    print(metric)
