from torch.nn import Module
from torch.nn import Parameter
from torch import tensor


class PetModel(Module):  # adapted from: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb
    def __init__(self):  # TODO: Implement the model (or use built model in segmentation_models_pytorch might be easier)
        super().__init__()

    def forward(self, x):
        return x
