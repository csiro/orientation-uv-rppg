import matplotlib.pyplot as plt
from pathlib import Path
from src.datamodules.datapipes import DatasetOperation
from abc import abstractmethod

from typing import *


class Plot(DatasetOperation):
    def __init__(self, name: str, root: Optional[str] = None, save: Optional[bool] = False, *args, **kwargs) -> None:
        super(Plot, self).__init__(*args, **kwargs)
        self.root = Path(root) if root is not None else Path(".").resolve().absolute()
        self.name = name
        self.save = save
        self.path = self.root.joinpath(self.name)

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        fig, ax = plt.subplots(*args, **kwargs)
        fig, ax = self.plot(fig, ax, data)
        if self.save: fig.savefig(self.path)

    @abstractmethod
    def plot(self, fig, ax, data: Dict[str, Any], *args, **kwargs) -> Tuple[Any, Any]:
        pass


'''
    print(idx, data["detections"].start, data["detections"].stop)
    
    for subkey in ["labels"]: #, "predictions"]:

        fn = DominantFrequency({}, "sps", [0.7,2.5], key=subkey)
        fn(data)

        signal = data[f"{subkey}"]
        frequencies = data[f"{subkey}_frequencies"]
        f_mask = (frequencies >= 0.7) & (frequencies <= 2.5)

        power_spectrum = data[f"{subkey}_power"]

        cont = ax.plot(signal, color="blue", label=f"{idx}")

        # cont[1].semilogy(frequencies[f_mask], power_spectrum[f_mask], label=f"{subkey} PSD")
        # cont[1].set_xlabel("Frequency [Hz]")
        # cont[1].set_ylabel("PSD [V**2/Hz]")
        # cont[1].grid(visible=True)
        # cont[1].legend(loc="best")

        # cont[2].plot(frequencies[f_mask], power_spectrum[f_mask], label=f"{subkey} PSD")
        # cont[2].set_xlabel("Frequency [Hz]")
        # cont[2].set_ylabel("PSD [V**2/Hz] (semilog)")
        # cont[2].grid(visible=True)
        # cont[2].legend(loc="best")

        artists.append(cont)

        # n_frames = data["frames"].shape[0]
        # ax[3].imshow(data["frames"].permute(0,2,3,1).numpy()[n_frames//2])
    if idx > 10:
        break

import matplotlib.animation as animation
ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=500)
ani.save(filename="signals.gif", writer="pillow")
'''