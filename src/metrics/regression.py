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
from torchmetrics.regression import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef

from typing import *
from torch import Tensor


class PearsonCorrCoefItems(PearsonCorrCoef):
    """
    """
    def __init__(self, *args, **kwargs) -> None:
        super(PearsonCorrCoefItems, self).__init__()

    def update(self, outputs, targets, *args, **kwargs) -> Any:
        """_summary_

        inputs to the functional will typically have a [B, N] format, where N is the
        number of outputs (items e.g. HR BPM)

        typically this will be 1 and hence [B, 1] so squeeze along last dim

        Args:
            outputs (_type_): _description_
            targets (_type_): _description_
        """
        if len(outputs.size()) > 1: outputs = outputs.squeeze(-1)
        if len(targets.size()) > 1: targets = targets.squeeze(-1)
        super(PearsonCorrCoefItems, self).update(outputs, targets, *args, **kwargs)

    def compute(self) -> Any:
        loss = super(PearsonCorrCoefItems, self).compute()
        loss[torch.isnan(loss)] = .0
        return loss


class AveragePearsonCoffCoef(Metric):
    """ NegativePearson Correlation Coefficient

    Calculate the Negative Pearson Correlation Coefficient (r) between the target and predicted signals.

    """
    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False # minimize the metric
    full_state_update: bool = False # accumulate

    def __init__(self, *args, **kwargs) -> None:
        super(AveragePearsonCoffCoef, self).__init__()
        self.add_state("loss", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("batch_size", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, predictions: Tensor, targets: Tensor, *args, **kwargs) -> None:
        """ Compute the loss for the provided batch and update the internal state.

        Args:
            predictions (Tensor): [N,T]
            targets (Tensor): [N,T]
        """
        # Compute the negative pearson correlation coefficient
        batch_losses = pearson_corrcoef(
            preds = predictions.permute(1,0),
            target = targets.permute(1,0)
        )

        # Update metric state
        self.loss += torch.sum(batch_losses)
        self.batch_size += predictions.size(0)

    def compute(self) -> Tensor:
        return self.loss / self.batch_size


class NegativePearsonCoffCoef(Metric):
    """ NegativePearson Correlation Coefficient

    Calculate the Negative Pearson Correlation Coefficient (r) between the target and predicted signals.

    L = 1 - sum((x_i - x_mu) * (y_i - y_u)) / sqrt(sum((x_i - x_mu)^2) * sum((y_i - y_mu)^2))

    L = 1 - covariance(X,Y) / (STD_X * STD_Y)

    rPPG Regression:
        Biases towards learning signals with a matching trend (correlation) to allow for accurate
        estimation of the peaks and troughs which is important for HR/HRV analysis.

    NOTE: Originally used in the `PhysNet` paper.

    """
    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False # minimize the metric
    full_state_update: bool = False # accumulate

    def __init__(self, *args, **kwargs) -> None:
        super(NegativePearsonCoffCoef, self).__init__()
        self.add_state("loss", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("batch_size", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, predictions: Tensor, targets: Tensor, *args, **kwargs) -> None:
        """ Compute the loss for the provided batch and update the internal state.

        Args:
            predictions (Tensor): [N,T]
            targets (Tensor): [N,T]
        """
        # Compute the negative pearson correlation coefficient
        batch_losses = 1 - pearson_corrcoef(
            preds = predictions.permute(1,0),
            target = targets.permute(1,0)
        )

        # Update metric state
        self.loss += torch.sum(batch_losses)
        self.batch_size += predictions.size(0)

    def compute(self) -> Tensor:
        return self.loss / self.batch_size