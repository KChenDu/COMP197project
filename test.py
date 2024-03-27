from argparse import ArgumentParser
from settings import OxfordIIITPet_DATA_ROOT, TESTING_BATCH_SIZE as BATCH_SIZE, DEVICE_COUNT, MODEL
from data.datasets import SimpleOxfordPetDataset
from torch.utils.data import DataLoader
from torch import load
from utils import Tester
from loguru import logger


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model_checkpoint')
    args = parser.parse_args()

    # Download the dataset if it does not exist
    SimpleOxfordPetDataset.download(OxfordIIITPet_DATA_ROOT)

    # Load the test dataset
    test_dataset = SimpleOxfordPetDataset(OxfordIIITPet_DATA_ROOT, "test")

    # Create the dataloader
    test_dataloader = DataLoader(test_dataset, BATCH_SIZE, num_workers=DEVICE_COUNT)

    # Import the model
    model = MODEL()
    # Load the model checkpoint
    model.load_state_dict(load(args.model_checkpoint)['model_state_dict'])

    tester = Tester()
    metric = tester.test(model, test_dataloader)
    logger.info(f"result: {metric}")
