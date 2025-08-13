'''
`update`: 
    Clear computed cache
    Call user-defined `update` 

`compute`:
    Synchronize metric states between processes (implication for training?)
    Reduce gathered metric states
    Call user defined `compute` method on gathered metric states
    Cache compute result
'''

import torch
from torchmetrics import Metric
from torch.nn.functional import cross_entropy

from typing import *


class CrossEntropy(Metric):
    """_summary_

    Args:
        Metric (_type_): _description_

    Returns:
        _type_: _description_
    """
    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False # minimize the metric
    full_state_update: bool = False # accumulate

    def __init__(self, *args, **kwargs):
        super(CrossEntropy, self).__init__()
        self.add_state("loss", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("batch_size", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, outputs, targets):
        assert outputs.size(0) == targets.size(0)
        self.batch_size += targets.size(0) # batch_size
        self.loss += cross_entropy(outputs, targets, reduction="sum") # sum over batch

    def compute(self):
        return self.loss / self.batch_size
