from enum import Enum
from pathlib import Path
from functools import partial
from src.datamodules.datasources.paths import Paths

from typing import *


class PURE_Subject(Enum):
    ALL = "*-*"


class PURE_VideoDevice(Enum):
    ECO274CVGE = "ECO274CVGE"


class PURE_PPGDevice(Enum):
    pass


class PURE_Scenario(Enum):
    """

    The details of the six scenarios are stated as follows:
        01 (steady): The subject was sitting still and looks directly into the camera avoiding head motion.
        02 (talking): Simulated video sequence, where the subjects were asked to talk while avoiding additional head motion. This setup equals a video conference situation in a real robot application.
        03 (slow translation): These sequences comprise head movements parallel to the camera plane. Therefore, the images recorded by the camera were displayed on screen and shown to the subjects. A moving rectangle of the size of the face was added to the image, and the subjects were asked to keep their face inside. The rectangle was moving horizontally at a controlled speed and with a predefined pattern, thus the sequences of all individuals are repeatable. The average speed was 7% of the face height per second, where the average face height was 100 pixels.
        04 (fast translation): This dataset has the same setup as slow translation, except twice the speed of the moving target.
        05 (small rotation): This setup comprises different targets that were placed at 35 cm around the camera. The subjects were told to look at these targets in a predefined sequence. They were asked to move not only there eyes but orient their head. See Fig. \ref{fig:setup} for an impression of the setup. The one minute sequence of the targets is shown in the little clock in the figure. Random times ensure that the motion artifacts are not periodically. Depending on the distance between the camera and the subject, that roughly varies between 1 m and 1.3 m, the head rotation angles are round about 20°.
        06 (medium rotation): These sequences had the same setup as for small rotation, but with targets placed 70 cm around the camera resulting in average head angle of 35°.
    
    """
    STEADY = "01"
    TALKING = "02"
    SLOW_TRANSLATION = "03"
    FAST_TRANSLATION = "04"
    SMALL_ROTATION = "05"
    MEDIUM_ROTATION = "06"


class PURE_SourceFile(Enum):
    # Video
    VIDEO = "video.avi"
    VIDEO_TIMESTAMPS = "video_timestamps.txt"

    # BVP
    BVP = "bvp.HDF5"
    BVP_RESAMPLED = "bvp_resampled.HDF5"

    # Metadata
    METADATA = "metadata.HDF5"


def list_pure_default_files(root: str) -> Dict[str, Any]:
    root = Path(str(root))
    identifier = root.name
    paths = {}
    paths["data"] = root.joinpath(f"{identifier}.json")
    paths["images"] = list(root.glob(f"{identifier}/*.png"))
    paths["images"] = sorted(paths["images"], key = lambda f: int(f.stem[len("Image"):])) # CRITICAL to ensure ordered
    return paths


class PURE_Default_Files(Paths):
    """ Provides a structured method for iterating over sets of  files.
    
    File structure for the  dataset is as follows:
        <root>/
            <subject_id>-<scenario_id>/
                <subject_id>-<scenario_id>.json # Data file
                <subject_id>-<scenario_id>/
                    Image<timestamp>.png # Video frames
    
    """
    def __init__(self, root: str, regex: str) -> None:
        process = partial(list_pure_default_files)
        super(PURE_Default_Files, self).__init__(root=root, regex=regex, process=process)


def list_pure_files(root: str) -> Dict[str, Any]:
    root = Path(str(root))
    identifier = root.name