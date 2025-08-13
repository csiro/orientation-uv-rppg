from typing import Any, Dict
import torch
import numpy as np
import torch.nn.functional as t_f
from src.datamodules.datapipes import DatasetOperation
from src.datamodules.datapipes.transform import Difference, Normalize, Standardize, NormalizedDifference

from typing import *
from numpy import ndarray


class SignalDifference(Difference):
    """ Compute the n-th order forward difference between frames along the time-axis.
    frame[i] = frame[i+1] - frame[i] : Applied n times recursively
    """
    def __init__(self, signal: str, *args, **kwargs) -> None:
        super(SignalDifference, self).__init__(key=signal, *args, **kwargs)


class NormalizeSignal(Normalize):
    """ Normalize the frames by minimum/maximum
    frame[i] = (frame[i] - min) / (max - min)
    """
    def __init__(self, signal: str, *args, **kwargs) -> None:
        super(NormalizeSignal, self).__init__(key=signal, *args, **kwargs)

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> Any:
        if self.mode == "minmax":
            minimum = torch.min(data[self.key].data)
            maximum = torch.max(data[self.key].data)
            data[self.key].data = (data[self.key].data - minimum) / (maximum - minimum)
        elif self.mode == "std":
            std = torch.std(data[self.key].data) # population correction
            data[self.key].data = data[self.key].data / std
        else:
            pass
        data[self.key].data[torch.isnan(data[self.key].data)] = .0


# class StandardizeSignal(Standardize):
#     """ Standardize the frames such that mean=0, var=1.
#     frame[i] = (frame[i] - mean) / std
#     """
#     def __init__(self, signal: str, *args, **kwargs) -> None:
#         super(StandardizeSignal, self).__init__(key=signal, *args, **kwargs)

class NormalizeSignal(DatasetOperation):
    """ Apply normalization to the signal with respect to the current sample.
    """ 
    def __init__(self, key: str, *args, **kwargs) -> None:
        super(NormalizeSignal, self).__init__(*args, **kwargs)
        self.key = key

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        signal = data[self.key].data
        signal = (signal - torch.mean(signal)) / torch.std(signal)
        data[self.key].data = signal


class AccumulateSignal(DatasetOperation):
    def __init__(self, key: str, *args, **kwargs) -> None:
        super(AccumulateSignal, self).__init__(*args, **kwargs)
        self.key = key

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        signal = data[self.key].data 
        signal = torch.cumsum(signal, dim=0)
        data[self.key].data = signal


class NormalizedSignalDifference(NormalizedDifference):
    """ Compute the normalized frame difference between frames.
    frame[i] = (frame[i+1] - frame[i]) / (frame[i+1] + frame[i])
    """
    def __init__(self, signal: str, *args, **kwargs) -> None:
        super(NormalizedSignalDifference, self).__init__(key=signal, *args, **kwargs)


class ResampleSignal(DatasetOperation):
    def __init__(self, 
        skey: str,
        fkey: str,
        length: Optional[int] = None, 
        mode: Optional[str] = "nearest", 
        target_key: Optional[str] = "frames",
        *args, **kwargs
    ) -> None:
        """_summary_

        Args:
            length (int): Target length of the re-sampled signal
        """
        super(ResampleSignal, self).__init__(*args, **kwargs)

        self.skey = skey
        self.fkey = fkey

        self.length = length
        self.mode = mode.lower()
        self.target_key = target_key
        assert self.mode in ["nearest", "linear", "bilinear", "bicubic"], f"provided mode {mode} not available"

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        """_summary_

        Args:
            inputs (Dict[str, Any]): 
            targets (Dict[str, Any]): `labels` of shape [N]
            metadata (Dict[str, Any]): _description_
        """
        # Set length to video length if undefined
        out_length = self.length if self.length is not None else data[self.fkey].data.shape[0]
        in_length = data[self.skey].data.shape[0]

        # Re-sample signal using different methods
        if self.mode in ["nearest", "linear"]:
            y_in = data[self.skey].data.unsqueeze(0).unsqueeze(0)
            y_out = t_f.interpolate(y_in, size=out_length, mode=self.mode)
            y_out = y_out.squeeze(0).squeeze(0)
            data[self.skey].data = y_out
            data[self.skey].attrs["sps"] = data[self.skey].attrs["sps"] * out_length / in_length

        elif self.mode in ["bilinear", "bicubic"]:
            y_in = data[self.skey].data.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            y_out = t_f.interpolate(y_in, size=out_length, mode=self.mode)
            y_out = y_out.squeeze(0).squeeze(0).squeeze(0)
            data[self.skey].data = torch.mean(y_out, dim=0)
            data[self.skey].attrs["sps"] = data[self.skey].attrs["sps"] * out_length / in_length
        else:
            pass
        

from scipy.sparse import diags

class TimeFIIRHighPassFilter(DatasetOperation):
    """ An advanced detrending method with application to HRV analysis. 

    IEEE Trans Biomed Eng. 2002 Feb;49(2):172-5. 
    Tarvainen MP, Ranta-Aho PO, Karjalainen PA. 
    doi: 10.1109/10.979357. 
    PMID: 12066885.

    An advanced, simple to use, detrending method to be used before heart rate variability 
    analysis (HRV) is presented. The method is based on smoothness priors approach and 
    operates like a time-varying finite-impulse response high-pass filter. The effect of the 
    detrending on time- and frequency-domain analysis of HRV is studied.

    Suitable for univariate timeseries e.g. BVP/PPG trace.

    https://github.com/ubicomplab/rPPG-Toolbox/blob/main/evaluation/post_process.py#L99
    https://github.com/phuselab/pyVHR/blob/master/pyVHR/BVP/filters.py

    Args:
        Transform (_type_): _description_
    """
    def __init__(self, signal: str, lmbda: float, *args, **kwargs) -> None:
        """_summary_

        Args:
            lmbda (float): Regularization parameters
        """
        super(TimeFIIRHighPassFilter, self).__init__(*args, **kwargs)
        self.lmbda = lmbda
        self.key = signal
        
    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        xs = data[self.key].data.numpy()

        if len(xs.shape) > 1:
            detrended_signals = np.array([self.functional(x) for x in xs]) 
        else:
            detrended_signals = self.functional(xs)
        
        data[self.key].data = torch.tensor(detrended_signals)

    def functional(self, x: ndarray) -> ndarray:
        #
        length = x.shape[0]

        #
        I = np.identity(length)
        D2 = diags(
            diagonals = [1, -2, 1],
            offsets = [0, 1, 2],
            shape = (length - 2, length)
        ).toarray()

        # 
        detrended_signal = np.dot(I - np.linalg.inv(I + self.lmbda**2 * np.dot(D2.T, D2)), x).copy()

        return detrended_signal



from scipy.signal import butter, filtfilt


class ButterworthFilter(DatasetOperation):
    def __init__(self, order: int, frequencies: List[float], ftype: str, signal: str, *args, **kwargs) -> None:
        super(ButterworthFilter, self).__init__(*args, **kwargs)
        self.order = order
        self.frequencies = frequencies
        self.ftype = ftype
        self.key = signal

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        # 
        xs = data[self.key].data.numpy()
        spss = data[self.key].attrs["sps"]

        if len(xs.shape) > 1:
            filtered_signals = np.array([self.functional(x, sps) for (x,sps) in zip(xs, spss)])
        else:
            filtered_signals = self.functional(xs, spss)

        # Save filtered signal
        data[self.key].data = torch.tensor(filtered_signals)


    def functional(self, x: ndarray, sps: float) -> ndarray:
        # Design a Butterworth filter
        b_numerator, a_demoninator = butter(
            N = self.order,
            Wn = self.frequencies, # fs specified, same units as fp
            btype = self.ftype,
            output = "ba",
            fs = sps
        )

        # Apply a forward and backward pass of the filter
        filtered_signal = filtfilt(
            b_numerator,
            a_demoninator,
            x.astype(np.float32)
        ).copy()
        
        return filtered_signal


class Derivative:
    pass


class Integral:
    pass


from src.datamodules.datamodels.timeseries import TimeseriesKeys

class ResampleToVideo(DatasetOperation):
    """
    """
    def __init__(self, fkey: str, skey: str, interpolate: Callable, use_timestamps: Optional[bool] = True, *args, **kwargs) -> None:
        super(ResampleToVideo, self).__init__(*args, **kwargs)
        self.fkey = fkey
        self.skey = skey
        self.interpolate = interpolate
        self.use_timestamps = use_timestamps
    
    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        # Timestamps : current
        has_timestamps = data[self.fkey].attrs[TimeseriesKeys.TIMESTAMPS.value] is not None
        use_timestamps = self.use_timestamps and has_timestamps

        if use_timestamps:
            ''' interpolate signal from uniformly sampled timestamps to gt video timestamps
            '''
            # Obtain information for interpolation
            video_timestamps = data[self.fkey].attrs[TimeseriesKeys.TIMESTAMPS.value]
            video_min_time = np.min(video_timestamps)
            video_max_time = np.max(video_timestamps)
            video_fps = data[self.fkey].fps
            
            # new timestamps (assume PPG to be uniformly sampled)
            signal_x = video_timestamps

            # Current timeseries
            signal_xp = np.linspace(video_min_time, video_max_time, data[self.skey].length)
            signal_fp = data[self.skey].data
        else:
            ''' interpolate signal from uniformly sampled timestamps to uniformly samples video
            timestamps (since we have no actual timestamps we assume a fixed sps)
            '''
            # Obtain from framerate
            video_length = data[self.fkey].data.shape[0]
            video_fps = data[self.fkey].fps
            signal_lenth = data[self.skey].data.shape[0]

            # new timestamps (choice doesn't really matter) : same start/stop just different lengths
            signal_x = np.linspace(1, signal_lenth, video_length)

            # current timeseries
            signal_xp = np.linspace(1, signal_lenth, signal_lenth)
            signal_fp = data[self.skey].data

        # interpolate
        interp_fn = self.interpolate(xp=signal_xp, yp=signal_fp)
        signal_f = interp_fn(x=signal_x)

        # Re-assign and update sampling rate
        data[self.skey].data = signal_f
        data[self.skey].attrs[TimeseriesKeys.SPS.value] = video_fps
        data[self.skey].attrs[TimeseriesKeys.TIMESTAMPS.value] = signal_x # may just be a linspace in the case of no timestamps


class LinearInterpolation:
    def __init__(self, xp: ndarray, yp: ndarray) -> None:
        self.xp = xp
        self.yp = yp

    def __call__(self, x: ndarray) -> ndarray:
        return np.interp(x=x, xp=self.xp, fp=self.yp)


from scipy.interpolate import CubicSpline

class CubicSplineInterpolation:
    def __init__(self, xp: ndarray, yp: ndarray) -> None:
        self.cs = CubicSpline(x=xp, y=yp)

    def __call__(self, x: ndarray) -> ndarray:
        return self.cs(x)


from scipy.interpolate import CubicHermiteSpline

class CubicHermiteSplineInterpolation:
    def __init__(self, xp: ndarray, yp: ndarray, dydx: Optional[ndarray] = None) -> None:
        if dydx is None: dydx = np.diff(xp)
        self.cs = CubicHermiteSpline(x=xp, y=yp, dydx=dydx)

    def __call__(self, x: ndarray) -> ndarray:
        return self.chs(x)


class ComputeSamplingRate(DatasetOperation):
    """ Compute the average sampling rate from time-stamps if available.

    This is typically useful during evaluation when you need to re-compute
    the sps for the entire video

    """
    def __init__(self, key: str, *args, **kwargs) -> None:
        super(ComputeSamplingRate, self).__init__(*args, **kwargs)
        self.key = key

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        # print(self.key, data[self.key].attrs)
        # Timestamps : current
        if TimeseriesKeys.TIMESTAMPS.value in data[self.key].attrs:
            has_timestamps = data[self.key].attrs[TimeseriesKeys.TIMESTAMPS.value] is not None
            
            if has_timestamps:
                timestamps = data[self.key].attrs[TimeseriesKeys.TIMESTAMPS.value]
                # timestamps = timestamps / 1e9

                dt = timestamps[-1] - timestamps[0]
                n_samples = timestamps.shape[0]
                sps = n_samples / dt

                data[self.key].attrs[TimeseriesKeys.SPS.value] = sps


        # data["predictions/signal"].attrs["sps"] = 30.0
        # data["targets/labels_unnormalized"].attrs["sps"] = 30.0