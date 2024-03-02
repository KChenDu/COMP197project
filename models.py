from torch.nn import Module


class ResUNet(Module):
    def __init__(self, init_ch: int = 32, num_levels: int = 3, out_ch: int = 1):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError
