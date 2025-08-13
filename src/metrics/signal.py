import torch
import numpy as np
from torchmetrics import Metric
from src.datamodules.datapipes.process.signal import DominantFrequency
from torchmetrics.regression import PearsonCorrCoef

from typing import *
from torch import Tensor


class SignalToNoiseRatio(Metric):
    """ SignalNoiseRatio

    Calculate the SNR as the ratio of the area under the curve of the frequency spectrum of the ground 
    truth HR frequency to the area under the curve of the remainder of the frequency bands.

    """
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True # maximize the metric
    full_state_update: bool = False # accumulate

    def __init__(self, harmonic_delta: float, wn: List[float], output_loc: str, target_loc: str, db: Optional[bool] = True, *args, **kwargs) -> None:
        super(SignalToNoiseRatio, self).__init__()
        self.add_state("loss", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("batch_size", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.harmonic_delta = harmonic_delta
        self.wn = wn
        self.db = db
        self.output_loc = output_loc
        self.target_loc = target_loc

    def update(self, outputs: Dict[str, Any], targets: Dict[str, Any], *args, **kwargs) -> None:
        # Obtain the first and second harmoics of the ground-truth frequency (targets)
        harmonic_1 = targets[self.target_loc].attrs["peak_hz"]
        harmonic_2 = 2 * harmonic_1

        # Compute FFT
        # NOTE: This should have already been done!
        f_signal = outputs[self.output_loc].attrs["frequencies"]
        pxx_signal = outputs[self.output_loc].attrs["power"]

        # Compute masks for frequency ranges of interest
        '''
        bin around harmonic 1
        bin around harmonic 2
        remainder of signal between cut-off frequencies
        '''
        mask_harmonic_1 = (f_signal >= (harmonic_1 - self.harmonic_delta)) & \
            (f_signal <= (harmonic_1 + self.harmonic_delta))
        mask_harmonic_2 = (f_signal >= (harmonic_2 - self.harmonic_delta)) & \
            (f_signal <= (harmonic_2 + self.harmonic_delta))
        mask_remainder = (f_signal >= self.wn[0]) & \
            (f_signal <= self.wn[1]) & \
            ~mask_harmonic_1 & \
            ~mask_harmonic_2

        # Select the corresponding values from the frequency power spectrum
        # NOTE: Since we are computing the relative is doesn't matter whether is absolute or density
        pwr_harmonic_1 = torch.sum(torch.masked_select(pxx_signal, mask_harmonic_1)) # across all dims
        pwr_harmonic_2 = torch.sum(torch.masked_select(pxx_signal, mask_harmonic_2))
        pwr_remainder = torch.sum(torch.masked_select(pxx_signal, mask_remainder))

        # Calculate the SNR as the ratio of the areas
        if pwr_remainder != .0:
            snr = (pwr_harmonic_1 + pwr_harmonic_2) / pwr_remainder
            if self.db: snr = 20 * torch.log10(snr)
        else:
            snr = (pwr_harmonic_1 + pwr_harmonic_2) * pwr_remainder

        # Update metric state
        self.loss += snr
        self.batch_size += f_signal.size(0)

    def compute(self) -> Tensor:
        return self.loss / self.batch_size
