import cv2
import h5py
import numpy as np
import subprocess
from tqdm import tqdm
from pathlib import Path
from src.datamodules.datapipes import DataOperation
from src.datamodules.datamodels.timeseries import TimeseriesKeys
from src.datamodules.datasources.paths.pure import PURE_SourceFile
from src.datamodules.datasources.models.pure import PURE_Default_Data
from src.datamodules.datasources.models.pure import PURE_Keys

from typing import *


class PURE_Formatter(DataOperation):
    """
    """
    def __init__(self,
        root: str,
        overwrite_video: Optional[bool] = False,
        overwrite_bvp: Optional[bool] = False,
        overwrite_metadata: Optional[bool] = False,
        *args, **kwargs
    ) -> None:
        super(PURE_Formatter, self).__init__(*args, **kwargs)
        
        self.root = Path(root)

        self.overwrite_video = overwrite_video
        self.overwrite_bvp = overwrite_bvp
        self.overwrite_metadata = overwrite_metadata

    def __call__(self, path: str) -> str:
        """ Format a given `PURE_Default_Data` sample into the processed format.

        We import from the following directory structure (default PURE):
            <root>/
                <subject_id>-<scenario_id>/ # this is the path provided
                    ...

        We export into the following directory structure:
            <root>/
                <subject_id>-<scenario_id>/
                    ...

        """
        # Logging
        with tqdm() as pbar:
            # Extract data from PURE
            name = Path(path).name # <subject_id>-<scenario_id>
            pbar.set_description(f"[{name}] Loading PURE data...")
            source : PURE_Default_Data = PURE_Default_Data(path)

            # Create output directory : <new_root>/<subject_id>-<scenario_id>
            root = self.root.joinpath(name)
            root.mkdir(exist_ok=True, parents=True)

            # Export Video Data
            pbar.set_description(f"[{name}] Exporting VIDEO data...")
            self.export_video(root, source)

            # Export BVP Data
            pbar.set_description(f"[{name}] Exporting BVP data...")
            self.export_bvp(root, source)

            # Export Metadata
            pbar.set_description(f"[{name}] Exporting METADATa data...")
            self.export_metadata(root, source)

            pbar.set_description(f"[{name}] Completed formatting.")

        return root

    def export_video(self, root: Path, source: PURE_Default_Data) -> None:
        """ PURE `video` exists as .PNG images.

        We will progressively write these images into a `video` container.

        NOTE: Associated with this is writing the exact timestamps.

        """
        # Frames
        video_frames : Generator = source.video_frames_iter
        height, width = source.video_height, source.video_width
        fps = source.average_frame_rate # only use for avg. in video container

        # Path
        video_filepath = root.joinpath(PURE_SourceFile.VIDEO.value)
        timestamp_filepath = root.joinpath(PURE_SourceFile.VIDEO_TIMESTAMPS.value)

        # Conditionally skip
        if video_filepath.exists():
            if not self.overwrite_video:
                return

        # Launch FFMPEG sub-process to write frames
        # NOTE: https://trac.ffmpeg.org/wiki/Encode/H.264#:~:text=The%20range%20of%20the%20CRF,sane%20range%20is%2017%E2%80%9328.
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo", # force format
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24", # openCV uses BGR format
            "-r", f"{fps}",
            "-i", "-", # input 
            "-an",
            "-vcodec", "libx264", # video codec
            "-qp", "0", # constant rate factor = lossless (caution using CRF=0 as this does not always correspond to lossless under different codecs)
            # "-b:v", "5000k", # video bitrate
            f"{video_filepath}"
        ]

        # Launch subprocess
        # TODO: Resolve issue around stdin/stderr buffer filling up when issues occur with writing video using ffmpeg (should clear or otherwise handle sterr buffer)
        # NOTE: Assign video buffer to be size of video (may be an issue with extremely large video files...)
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE) #, stderr=subprocess.PIPE) # NOTE: We removed: stderr=subprocess.PIPE 

        # Process frames
        for idx, frame in tqdm(enumerate(video_frames), total=source.video_length):
            # Convert frames to OpenCV compatible format
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Write serialized frame to container via. ffmpeg sub-process
            proc.stdin.write(frame.tobytes())
        
        # Close process
        proc.stdin.close()
        # proc.stderr.close()
        proc.wait()

        # Load video timestamps (ns)
        video_timestamps : ndarray = source.video_timestamps

        # Write video timestamps
        np.savetxt(timestamp_filepath, video_timestamps)

    def export_bvp(self, root: Path, source: PURE_Default_Data) -> None:
        # Path
        filepath = root.joinpath(PURE_SourceFile.BVP.value)

        # Conditionally skip
        if filepath.exists():
            if not self.overwrite_bvp:
                return
                
        # Write to file
        with h5py.File(filepath, "w") as fp:
            ds = fp.create_dataset("data", data=source.bvp_samples)
            ds.attrs.create(TimeseriesKeys.TIMESTAMPS.value, source.sample_timestamps)


    def export_metadata(self, root: Path, source: PURE_Default_Data) -> None:
        # Path
        filepath = root.joinpath(PURE_SourceFile.METADATA.value)
        
        # Conditionally skip
        if filepath.exists():
            if not self.overwrite_metadata:
                return

        # Construct and write metadata
        attrs = {
            PURE_Keys.SUBJECT.value: source.subject,
            PURE_Keys.SCENARIO.value: source.scenario
        }
        
        # Write to HDF5 container
        with h5py.File(filepath, "w") as fp:
            for key, val in attrs.items():
                fp.attrs.create(key, val)


import numpy as np
from src.datamodules.datamodels.timeseries import TimeseriesKeys
from src.datamodules.datapipes import DataOperation, DatasetOperation


class PURE_ExportBVP(DatasetOperation):
    '''
    want timestamps:
        min(bvp) < min(vid)
        max(bvp) > max(vid)

    find the union of the domains of the timestamps
        interpolate the bvp to the video timestamps

    OR:

    assume they're sufficiently synchronized such that we don't need to synchronize
    ...may not always be the case

    then just resample to video gts

    '''
    def __init__(self, key: str, filename: str, *args, **kwargs) -> None:
        super(PURE_ExportBVP, self).__init__(*args, **kwargs)
        self.key = key
        self.filename = filename

    def apply(self, data: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        # Reference existing items
        bvp = data[self.key]

        # Extract root
        root = bvp.attrs["root"]
        filepath = root.joinpath(self.filename)

        # Write to file
        with h5py.File(filepath, "w") as fp:
            ds = fp.create_dataset("data", data=bvp.data)
            ds.attrs.create(TimeseriesKeys.TIMESTAMPS.value, 1e9 * bvp.attrs[TimeseriesKeys.TIMESTAMPS.value])