from torch import Tensor, sum
import torch


def dice_score(ps: Tensor, ts: Tensor, eps: float = 1e-7) -> Tensor:
    assert ps.ndim == 4 and ts.ndim == 4 and ps.size() == ts.size()
    numerator = sum(ts * ps, dim=(1, 2, 3)) * 2. + eps
    denominator = sum(ts, dim=(1, 2, 3)) + sum(ps, dim=(1, 2, 3)) + eps
    return numerator / denominator


def dice_loss(ps: Tensor, ts: Tensor) -> Tensor:
    assert ps.ndim == 4 and ts.ndim == 4 and ps.size() == ts.size()
    return 1. - dice_score(ps, ts)


def dice_binary(ps: Tensor, ts: Tensor) -> Tensor:
    assert ps.ndim == 4 and ts.ndim == 4 and ps.size() == ts.size()
    ps = (ps >= .5).type_as(ps)
    ts = (ts >= .5).type_as(ts)
    return dice_score(ps, ts)

def accuracy(ps: Tensor, ts: Tensor) -> Tensor:
    '''
    Calculate the accuracy of the model.
    
    Args:
    ps: The predicted values.
    ts: The target values.
    
    Returns:
    The accuracy of the model.
    '''
    assert ps.ndim == 4 and ts.ndim == 4 and ps.size() == ts.size()
    ps = torch.round(ps).type(torch.int)
    ts = torch.round(ts).type(torch.int)
    eqs = (ps == ts).type_as(ps)
    return sum(eqs, dim=(1, 2, 3)) / ts[0].numel()