# import matplotlib.pyplot as plt
# from src.datamodules.datapipes.visualise import Plot

# from typing import *


# class PeriodogramPlot(Plot):
#     def __init__(self, subkey: str, *args, **kwargs) -> None:
#         super(PeriodogramPlot, self).__init__(*args, **kwargs)
#         self.subkey = subkey

#     def plot(self, fig, ax, data: Dict[str, Any], *args, **kwargs) -> Tuple[Any, Any]:
#         frequencies = data[f"{self.subkey}_frequencies"]
#         power_spectrum = data[f"{self.subkey}_power"]
#         ax.semilogy(frequencies, power_spectrum, label="Power Spectrum Density")
#         ax.title(f"{self.subkey} Power Spectrum Density")
#         ax.xlabel("Frequency [Hz]")
#         ax.ylabel("PSD [V**2/Hz]")
#         ax.grid(visible=True)
#         ax.legend(loc="best")
#         return fig, ax


# class TimeseriesPlot(Plot):
#     def __init__(self, subkey: str, *args, **kwargs) -> None:
#         super(PeriodogramPlot, self).__init__(*args, **kwargs)
#         self.subkey = subkey

#     def plot(self, fig, ax, data: Dict[str, Any], *args, **kwargs) -> Tuple[Any, Any]:
#         labels = data[f"labels"]
#         predictions = data[f"predictions"]
#         ax.plot(labels, label=f"{self.subkey}_labels")
#         ax.plot(predictions, label=f"{self.subkey}_predictions")
#         ax.grid(visible=True)
#         ax.legend(loc="best")
#         return fig, ax