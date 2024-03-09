from torch.nn import Module
from torch.nn import Parameter
from torch import tensor


class PetModel(Module):  # adapted from: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb
    def __init__(self):
        super().__init__()
        self.w = Parameter(tensor([1., 2., 3.]))

    def forward(self, x):
        return x
