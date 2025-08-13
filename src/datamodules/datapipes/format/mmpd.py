import cv2
import h5py
import subprocess
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from src.datamodules.datapipes import DataOperation
from src.datamodules.datasources.paths.mmpd import MMPD_SourceFile
from src.datamodules.datasources.files.mmpd import MMPD_MATLAB, MMPD_Keys
from src.datamodules.datasources.models.mmpd import MMPD_DatasetModel_Raw

from typing import *


class MMPD_Formatter(DataOperation):
    """ Default MMPD file exists as a monolithic MATLAB file which does NOT support lazy loading
    or sliced IO access.

    Hence we separate our the components of the dataset during formatting e.g. video, timeseries
    and metadata into a more suitable format.

    All dataset file containers should be self descriptive. We use hence typically use video containers
    for video data and .HDF5 files for all other data.

    # TODO: Implement configuration defined Video Writer.
    
    """
    def __init__(self, 
        root: str, 
        overwrite_video: Optional[bool] = False,
        overwrite_bvp: Optional[bool] = False,
        overwrite_metadata: Optional[bool] = False,
        export_as_frames: Optional[bool] = False,
        use_ffmpeg: Optional[bool] = True,
        *args, **kwargs
    ) -> None:
        super(MMPD_Formatter, self).__init__(*args, **kwargs)
        self.root = Path(root)
        self.root.mkdir(exist_ok=True, parents=True)

        self.overwrite_video = overwrite_video
        self.overwrite_bvp = overwrite_bvp
        self.overwrite_metadata = overwrite_metadata

        self.export_as_frames = export_as_frames

        self.use_ffmpeg = use_ffmpeg

    def __call__(self, path: str) -> str:
        # Description
        name = Path(path).name

        # Logging
        with tqdm() as pbar:
            # Extract data from MMPD MATLAB file.
            pbar.set_description(f"[{name}] Loading MMPD data...")
            data = MMPD_MATLAB(path).data()
            data = MMPD_DatasetModel_Raw(data)

            # Create destination directory
            root = self.root.joinpath(f"p{data.subject}_{data.clip}")
            root.mkdir(exist_ok=True, parents=True)

            # Export Video Data
            pbar.set_description(f"[{name}] Exporting VIDEO...")
            self.export_video(root, data)

            # Export BVP Data
            pbar.set_description(f"[{name}] Exporting TIMESERIES...")
            self.export_bvp(root, data)

            # Export Metadata
            pbar.set_description(f"[{name}] Exporting METADATA...")
            self.export_metadata(root, data)

            pbar.set_description(f"[{name}] Completed formatting.")

        return root


    def export_video(self, root: Path, data: MMPD_DatasetModel_Raw) -> None:
        """

        NOTE: In formatting the raw data we introduce quantization error; fp32 -> uint8
        
        """
        # Extract frames
        frames = data.video
        height, width = frames.shape[1], frames.shape[2] # [1800, 320, 240, 3]

        # Frames
        if self.export_as_frames:
            # Update root
            root = root.joinpath("frames")
            root.mkdir(exist_ok=True, parents=True)

            # Process frames
            for idx, frame in tqdm(enumerate(frames), desc=f"[{root.name}]"):
                # Filepath
                filename : str = filename_frame(idx)
                filepath = root.joinpath(filename) # e.g. <root>/frames/frame_00000001.png # allow for sorting

                # Conditionally skip
                if filepath.exists():
                    if not self.overwrite_video:
                        continue

                # Convert frames to PIL format : [0-255] [HWC] uint8
                frame = (2**8 - 1) * frame # [0-1] > [0-255]
                frame = frame.astype(np.uint8) # [FP32] > [UINT8]
                frame = Image.fromarray(frame)

                # Save image in lossless format
                frame.save(filepath)
        else:
            # Path
            path = root.joinpath(MMPD_SourceFile.VIDEO.value)

            # Conditionally skip
            if path.exists():
                if not self.overwrite_video:
                    return

            if self.use_ffmpeg:
                 # FFMPEG - command for consumer
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-f", "rawvideo", # force format
                    "-vcodec", "rawvideo",
                    "-s", f"{width}x{height}",
                    "-pix_fmt", "bgr24", # openCV uses BGR format
                    "-r", f"{data.video_fps}",
                    "-i", "-", # input 
                    "-an",
                    "-vcodec", "libx264", # video codec
                    "-qp", "0", # constant rate factor = lossless
                    "-b:v", "5000k", # video bitrate
                    f"{path}"
                ]

                # Launch subprocess
                proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)#, stderr=subprocess.PIPE)

                # Process frames
                for idx, frame in tqdm(enumerate(frames), desc=f"[{root.name}]"):
                    # Convert frames to OpenCV compatible format (U8 || U16)
                    frame = (2**8 - 1) * frame # [0-1] to [0-255]
                    frame = frame.astype(np.uint8) # [FP32] to [UINT8] # TODO: Avoid lossy conversion
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # [RGB] to [BGR]

                    # Write frames to container via pipe to FFMPEG subprocess
                    proc.stdin.write(frame.tobytes())

                # Close process
                proc.stdin.close()
                # proc.stderr.close()
                proc.wait()
                
            else:
                # Create VideoWriter
                writer = cv2.VideoWriter(
                    filename = str(path),
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG'),
                    fps  = data.video_fps,
                    frameSize = [width, height]
                )

                # Process frames
                for idx, frame in tqdm(enumerate(frames), desc=f"[{root.name}]"):
                    # Convert frames to OpenCV compatible format (U8 || U16)
                    frame = (2**8 - 1) * frame # [0-1] to [0-255]
                    frame = frame.astype(np.uint8) # [FP32] to [UINT8] # TODO: Avoid lossy conversion
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # [RGB] to [BGR]

                    # Write frames to container
                    writer.write(frame)

                # Release writer
                writer.release()

                

    def export_bvp(self, root: Path, data: MMPD_DatasetModel_Raw) -> None:
        """
        """
        # Path
        path = root.joinpath(MMPD_SourceFile.BVP.value)

        # Conditionally skip
        if path.exists():
            if not self.overwrite_bvp:
                return

        # Create .HDF5 file
        with h5py.File(path, "w") as fp:
            # Write BVP data to HDF5 dataset
            ds = fp.create_dataset(f"{MMPD_Keys.GT_BVP.value}", data=data.gt_bvp)

            # Create associated attributes
            for key in [
                MMPD_Keys.GT_BVP_SPS
            ]:
                ds.attrs.create(key.value, data.data[key.value]) # TODO: Should only interface via. model properties.

    def export_metadata(self, root: Path, data: MMPD_DatasetModel_Raw) -> None:
        """
        """
        # Path
        path = root.joinpath(MMPD_SourceFile.METADATA.value)

        # Conditionally skip
        if path.exists():
            if not self.overwrite_metadata:
                return

        # Create .HDF5 file
        with h5py.File(path, "w") as fp:
            # Create associated attributes
            for key in [
                MMPD_Keys.SUBJECT,
                MMPD_Keys.CLIP,
                MMPD_Keys.LIGHT,
                MMPD_Keys.MOTION,
                MMPD_Keys.EXERCISE,
                MMPD_Keys.SKIN_TONE,
                MMPD_Keys.GENDER,
                MMPD_Keys.GLASSES,
                MMPD_Keys.HAIR_COVER,
                MMPD_Keys.MAKEUP
            ]:
                fp.attrs.create(key.value, data.data[key.value])


def filename_frame(frame_index: int) -> str:
    return f"frame_{str(frame_index).zfill(8)}.png"
