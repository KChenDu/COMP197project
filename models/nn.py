from torch.nn import Module


class PetModel(Module):  # adapted from: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
