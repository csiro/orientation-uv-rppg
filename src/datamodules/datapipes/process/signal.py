'''
Signal Procesing

De-trending
Frequency-domain
Time-domain

'''
import torch
import scipy
import numpy as np
import torch.nn.functional as t_f
from functools import partial
# from scipy.signal import find_peaks, periodogram
from src.datamodules.datamodels import DataModel
from src.datamodules.datapipes import DatasetOperation

from typing import *
from numpy import ndarray


class InPlaceScipyTransform(DatasetOperation):
    """ General interface for applying SciPy processes to a signal.

    Example::
        >>> filter = partial(scipy.signal.find_peaks, **{...kwargs for find_peaks...})
        >>> transform = ScipyTransform(filter, "labels")
        >>> data = dataset.__getitem__(idx)
        >>> transform(data) # perform in-place find-peaks

    Args:
        Transform (_type_): _description_
    """
    def __init__(self, process: Callable, key: Optional[str] = "labels", *args, **kwargs) -> None:
        super(InPlaceScipyTransform, self).__init__(*args, **kwargs)
        self.process = process
        self.key = key

    def apply(self, data: Dict[str, Union[DataModel, Any]], *args, **kwargs) -> None:
        data[self.key].data = torch.tensor(self.process(data[self.key].data).toarray())


class AveragePeakFrequency(DatasetOperation):
    """ Find the peaks in a signal based on the properties of the peaks such as: height, threshold,
    distance, prominence, width, etc.

    Returns indices of the peaks in the input array which satisfy the given conditions, and a dictionary
    describing the properties of the peaks.

    Given we're not dealing with plateau style peaks we can use this, but consider either filtering
    before using this for such signals, or applying a wavelet transform.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

    Args:
        ScipyTransform (_type_): _description_
    """
    def __init__(self, signal: str, find_peaks: Optional[Dict[str, Any]] = {}, *args, **kwargs) -> None:
        """ Uses `scipy.signal.find_peaks` to find peaks in the signal.

        Args:
            process_kwargs (Dict[str, Any]): Keyword arguments to provide to the peak detection algorithm
            key (str): `DataModel` in the `data` to compute for and update.
        """
        super(AveragePeakFrequency, self).__init__(*args, **kwargs)
        self.find_peaks = partial(scipy.signal.find_peaks, **find_peaks)
        self.key = signal

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        """_summary_

        avg_didx / idxs/sec = avg_didx * sec/idx = avg_dt
        idxs/sec / avg_didx = 1/avg_dt = avg_hz
        b/s * 60s/m = 60*b/m

        Args:
            data (Dict[str, Any]): Appends data["hr_bpm"] (Tensor[1]) to the dictionary.
        """
        # Extract data points
        xs = data[self.key].data.numpy()
        spss = data[self.key].sps

        # Calculate average frequency of the peaks
        if len(data[self.key].data.size()) > 1:
            peaks_avg_hz = np.array([self.transform(x, sps) for (x,sps) in zip(xs, spss)])
        else:
            peaks_avg_hz = self.transform(xs, spss)

        # Update `DataModel` to specified units
        data[self.key].attrs["peak_bpm"] = 60.0 * torch.tensor(peaks_avg_hz)
        data[self.key].attrs["peak_hz"] = torch.tensor(peaks_avg_hz)

    def transform(self, x: ndarray, sps: float) -> float:
        # Calculate average frequency of the peaks
        peak_idxs, _ = self.find_peaks(x) # [idx0, idxN, ..., idxM]
        peaks_avg_idx = np.mean(np.diff(peak_idxs)) # [idxN-idx0, ...] = [dIdx0, ..., dIdxN] -> avg_didx
        peaks_avg_hz = sps / peaks_avg_idx # avg_didx / (idxs/second) = avg_per_sec
        return peaks_avg_hz



class DominantFrequency(DatasetOperation):
    """ Compute the dominant frequency of a signal using the Fast Fourier Transform within a certain frequency range.

    Args:
        ScipyTransform (_type_): _description_
    """
    def __init__(self, signal: str, wn: List[float], periodogram: Optional[Dict[str, Any]] = {}, quick_fft: Optional[bool] = True, length_multiple: Optional[int] = 1, *args, **kwargs) -> None: 
        """_summary_

        Args:
            fn_args (Dict[str, Any]): periodogram kwargs
            key (str): `DataModel` in the `data` to compute for and update.
            wn (List[float]): cutoff frequencies
        """
        super(DominantFrequency, self).__init__(*args, **kwargs)
        self.periodogram = partial(scipy.signal.periodogram, **periodogram)
        self.key = signal
        self.wn = wn
        self.quick_fft = quick_fft
        self.length_multiple = length_multiple
        
        assert len(self.wn) == 2, f"Must provide a lower and upper cut-off freq [w_lower, w_upper]"

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        """_summary_

        NOTE: We do NOT de-trend the signal. Rely

        Args:
            data (Dict[str, Any]): _description_

        Returns:
            _type_: _description_
        """
        # Input
        xs = data[self.key].data.numpy()
        spss = data[self.key].sps
        
        # 
        if len(xs.shape) > 1: # [B, L]
            outs = [self.functional(x, sps) for (x,sps) in zip(xs, spss)]
            f_signals = np.array([out[0] for out in outs])
            pxx_signals = np.array([out[1] for out in outs])
            f_peak_hzs = np.array([out[2] for out in outs])
        else: # [L]
            f_signals, pxx_signals, f_peak_hzs = self.functional(xs, spss)
        
        # Update `DataModel` to specified units
        data[self.key].attrs["frequencies"] = torch.tensor(f_signals) # [L]
        data[self.key].attrs["power"] = torch.tensor(pxx_signals) # [B, L]
        data[self.key].attrs["peak_bpm"] = 60.0 * torch.tensor(f_peak_hzs)
        data[self.key].attrs["peak_hz"] = torch.tensor(f_peak_hzs)

    def functional(self, x: ndarray, sps: float) -> Tuple[ndarray, ndarray, float]:
        # Length of the FFT (see discussion on spectral resolution)
        if self.quick_fft:
            length = int(x.shape[0])
            length = 1 if length == 0 else 2 ** (length - 1).bit_length()
            length = self.length_multiple * length
        else:
            length = None

        # Compute FFT of the signal
        # https://dsp.stackexchange.com/questions/10043/how-important-is-it-to-use-power-of-2-when-using-fft
        f_signal, pxx_signal = self.periodogram(
            x,
            fs = sps,
            window = "boxcar",
            nfft = length, # FFT window size
            detrend = False
        )

        # Mask signal
        f_mask = (f_signal >= self.wn[0]) & (f_signal <= self.wn[1])
        f_idxs = np.argwhere(f_mask)
        f_masked = f_signal[f_idxs]
        pxx_masked = pxx_signal[f_idxs]

        # Compute dominant frequency
        pxx_max_idx = np.argmax(pxx_masked)
        f_peak_hz = f_masked[pxx_max_idx]

        return f_signal, pxx_signal, f_peak_hz
