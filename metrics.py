from torch import Tensor, sum, round, int8
from segmentation_models_pytorch.metrics import get_stats, accuracy


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


def mean_binary_accuracy(ps: Tensor, ts: Tensor) -> Tensor:
    '''
    Calculate the accuracy of the model.
    
    Args:
    ps: The predicted values.
    ts: The target values.
    
    Returns:
    The accuracy of the model.
    '''
    ps = round(ps).type(int8)
    ts = round(ts).type(int8)
    tp, fp, fn, tn = get_stats(ps, ts, 'binary', threshold=.5)
    
    return accuracy(tp, fp, fn, tn, reduction='macro')
    # ps = torch.round(ps).type(torch.int)
    # ts = torch.round(ts).type(torch.int)
    # eqs = (ps == ts).type_as(ps)
    # return sum(eqs, dim=(1, 2, 3)) / ts[0].numel()

def binary_accuracies(ps: Tensor, ts: Tensor) -> Tensor:
    '''
    Calculate the accuracy of the model.
    
    Args:
    ps: The predicted values.
    ts: The target values.
    
    Returns:
    The accuracy of the model.
    '''
    ps = round(ps).type(int8)
    ts = round(ts).type(int8)
    tp, fp, fn, tn = get_stats(ps, ts, 'binary', threshold=.5)
    
    accuracies = Tensor(len(ps))
    for i in range(len(ps)):
        accuracies[i] = accuracy(tp[i], fp[i], fn[i], tn[i], reduction='macro')
    return accuracies