import torch
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from src.datamodules.datapipes import DataOperation

from typing import *

sns.set_theme(style="darkgrid")


class ExportPPGResults(DataOperation):
    def __init__(self, root: str, *args, **kwargs) -> None:
        super(ExportPPGResults, self).__init__(*args, **kwargs)
        self.root = Path(root).absolute().resolve()

    def __call__(self, videos: Dict[str, Dict[str, Any]], *args, **kwargs) -> None:
        for key, val in videos.items():
            src = val["source/sample"].attrs
            dataset_name = src["dataset_name"]
            model_name = src["model_name"]
            model_uuid = src["model_uuid"]

            out_unproc = val["predictions/signal_copy"]
            out_proc = val["predictions/signal"]
            tgt_unproc = val["targets/labels_unnormalized_copy"]
            tgt_proc = val["targets/labels_unnormalized"]

            sps = out_proc.attrs["sps"]

            fig, ax = plt.subplots(
                figsize = (16,24),
                nrows=3, ncols=1,
                # tight_layout=False,
                # constrained_layout=True
            )
            fig.suptitle(f"{model_name} ({model_uuid})\n{dataset_name}: {key}", fontsize=24)

            # # Plot un-processed PPG signal
            sns.lineplot(tgt_unproc.data, label="Ground-truth PPG", alpha=0.75, ax=ax[0])
            sns.lineplot(out_unproc.data, label="Predicted PPG", alpha=0.75, ax=ax[0])
            ax[0].set_xlim((0, out_unproc.data.size(0)))
            ax[0].grid(True, alpha=0.50)
            ax[0].set_title(f"Un-processed PPG Waveforms")
            ax[0].legend(loc="upper right")
            ax[0].set_ylabel("BVP PPG")
            ax[0].set_xlabel(f"Timesteps [SPS={sps}]")

            # # Plot processed PPG signal
            sns.lineplot(tgt_proc.data, label="Ground-truth PPG", alpha=0.75, ax=ax[1])
            sns.lineplot(out_proc.data, label="Predicted PPG", alpha=0.75, ax=ax[1])
            ax[1].set_xlim((0, out_unproc.data.size(0)))
            ax[1].grid(True, alpha=0.50)
            ax[1].set_title(f"Processed PPG Waveforms")
            ax[1].legend(loc="upper right")
            ax[1].set_ylabel("BVP PPG")
            ax[1].set_xlabel(f"Timesteps [SPS={sps}]")

            # Plot frequency power-spectrum
            out_f_signal = out_proc.attrs["frequencies"]
            out_pxx_signal = out_proc.attrs["power"]
            tgt_f_signal = tgt_proc.attrs["frequencies"]
            tgt_pxx_signal = tgt_proc.attrs["power"]

            out_peak_bpm = out_proc.attrs["peak_bpm"].item()
            tgt_peak_bpm = out_proc.attrs["peak_bpm"].item()

            sns.lineplot(x=60*out_f_signal, y=out_pxx_signal, alpha=0.75, label="Ground-truth Power Spectrum", ax=ax[2])
            g = sns.lineplot(x=60*tgt_f_signal, y=tgt_pxx_signal, alpha=0.75, label="Predicted Power Spectrum", ax=ax[2])
            g.set_yscale("log")
            ax[2].axvline(tgt_peak_bpm, alpha=0.75, color="red", linestyle="--", label=f"Ground-truth HP BPM: {tgt_peak_bpm:.3f}")
            ax[2].axvline(out_peak_bpm, alpha=0.75, color="black", linestyle="--", label=f"Predicted HR BPM: {out_peak_bpm:.3f}")
            ax[2].set_xlim(left=0, right=torch.max(60*tgt_f_signal).item())
            ax[2].set_ylim(bottom=1e-7, top=1e3)
            ax[2].grid(True, alpha=0.50)
            ax[2].set_title("Processed PPG Power Spectrum")
            ax[2].legend(loc="upper right")
            ax[2].set_ylabel("Power (V**2/Hz)")
            ax[2].set_xlabel(f"Frequency (BPM)")

            fig.savefig(self.root.joinpath(f"{model_name}_{model_uuid}_{dataset_name}_{key}"))
            plt.close(fig)

        return videos