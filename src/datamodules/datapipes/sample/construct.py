import cv2
import h5py
import numpy as np
from tqdm import tqdm
from uuid import uuid4
from pathlib import Path

from src.datamodules.datamodels import calculate_length
from src.datamodules.datapipes import DataOperation
from src.datamodules.datamodels import DetectionModel
from src.datamodules.datasources import DatasetSample
from src.datamodules.datasources.mmpd import MMPD_DatasetSample
from src.datamodules.datasources.files.detections import DetectionsFile

from typing import *
from numpy import ndarray



class GenerateFrameSlices(DataOperation):
    """ Generate chunks/slices of video frames and store in a file reference, and link
    through External Soft-links to the original datasets.

    """
    def __init__(self,
        window: int, 
        start: Optional[int] = 0, 
        stride: Optional[int] = None, 
        stop: Optional[int] = None,
        video_kwargs: Optional[str] = {},
        *args, **kwargs
    ) -> None:
        super(GenerateFrameSlices, self).__init__(*args, **kwargs)

        # Frames
        self.video_kwargs = video_kwargs

        # Convolution
        self.window = window
        self.start = start
        self.stride = stride if stride is not None else self.window
        self.stop = stop

        if self.start is not None: assert self.start >= 0, f"Slice index must be > 0, TBD handle underflow later..."
        # if self.stop is not None: assert self.stop >= 0, f"Slice index must be > 0, TBD handle underflow later..."

    def __call__(self, sources: Iterable[DatasetSample]) -> Dict[str, Any]:
        """ Create the minimum necessary information to reconstruct dynamically sliced `DatasetSample`s at 
        run-time.

        NOTE: Involves creating a temporary artifact which stores the start and stop index (int) along with
        the source

        """
        # Just store in a dict
        samples : Dict[int, DatasetSample] = {}
        num_samples = 0

        with tqdm() as pbar:

            # Create slices for each root (sample)
            for source in sources:
                pbar.set_description(f"[{source.path}] Preparing source...")

                # Pipe from the input file
                video_length = source.video(**self.video_kwargs).video_length

                # Bound start/stop values
                start = self.start if self.start is not None else 0
                stop = self.stop if self.stop is not None else video_length
                if stop < 0: stop = video_length - stop # underflow

                # Create contiguous sections for windows
                if source.detections_filename is not None:
                    # Bound via. detections
                    detections : DetectionModel = source.detections().data()
                    
                    # Extract frame indexes to convolve over
                    frame_indexes : ndarray = detections.frame_indexes(detections.data)

                    # Define HARD bounds to use for frames
                    hard_start = min(frame_indexes) if min(frame_indexes) > start else start
                    hard_stop = max(frame_indexes) if max(frame_indexes) < stop else stop

                    # Compute index differences (gaps in the indexes)
                    index_diffs = np.diff(frame_indexes)
                    index_diff_indexes = np.where(index_diffs > 1)[0].tolist()

                    # Define contiguous segments for convolution (extracting windows)
                    if len(index_diff_indexes):
                        # Define as [hard_star, start_gap1, stop_gap1, ..., hard_stop]
                        section_indexes = [hard_start]
                        for idx in index_diff_indexes:
                            section_indexes.append(frame_indexes[idx])
                            section_indexes.append(frame_indexes[idx+1])
                        section_indexes.append(hard_stop)

                    else: # No gaps
                        section_indexes = [hard_start, hard_stop]

                else: # No detections (use frame limits)
                    section_indexes = [start, stop]

                #
                pbar.set_description(f"[{source.path}] Generating slices...")

                # Perform convolution over frame-contiguous segments and extract windows
                key_idx = 0
                used_frames = 0
                for idx in range(len(section_indexes) - 1):
                    if idx % 2 != 0: continue # skip every second (skip the gaps)

                    # Define bounds of section
                    section_start = section_indexes[idx]
                    section_stop = section_indexes[idx + 1]
                    used_frames += (section_stop - section_start)

                    # Convolve over window (define start:stop)
                    key_idx, samples, num_samples = self.convolve_window(
                        key_idx, 
                        source, 
                        samples,
                        video_length, 
                        section_start, 
                        section_stop,
                        pbar,
                        num_samples
                    )

                # if used_frames < input_interface.frames.length:
                #     log.debug(f"{path.stem}: Using {used_frames} out of {input_interface.frames.length}")

        # return self.context(memory_output_file, mode="r", rdcc_nbytes=0) # return file context to be used
        return samples

    def convolve_window(self, 
        key_idx : int, 
        source : DatasetSample, 
        samples : Dict,
        video_length : int,
        start : int,
        stop: int,
        pbar: Any,
        num_samples: int
    ) -> None:
        """
        """
        # Define maximum length with current start/stop
        length = calculate_length(video_length, start, stop)

        # Convolve over frames based on stride
        for idx in range((length + self.stride - self.window) // self.stride):
            #
            pbar.update(1)
            num_samples += 1

            idx_name = len(samples.keys()) #f"{source.path.stem}_{key_idx}"
            key_idx += 1

            # Arguments
            kwargs = {
                "source": source.path, # actually the root for datasets
                "start": start + self.stride * idx,
                "stop": start + self.stride * idx + self.window,
                "detections": source.detections_filename
            }

            # Create sample
            samples[idx_name] = type(source)(**kwargs)
        
        return key_idx, samples, num_samples

import cv2
import h5py
import torch
from src.datamodules.datamodels.frames import VideoFramesModel
from src.datamodules.datamodels.landmarks import LandmarksModel
# from src.datamodules.datasources.


class ProcessDatasetSamples(DataOperation):
    def __init__(self,
        process: Callable,
        filenames: Dict[str, Any],
        use_detections: Optional[bool] = True,
        dkey: Optional[str] = None,
        fkey: Optional[str] = None,
        *args, **kwargs
    ) -> None:
        super(ProcessDatasetSamples, self).__init__(*args, **kwargs)
        self.process = process
        self.filenames = filenames
        self.use_detections = use_detections
        self.dkey = dkey
        self.fkey = fkey

    def __call__(self, sources: Iterable[DatasetSample]) -> Dict[str, Any]:
        for source in sources: # process sources 1-by-1 (nicer to process file-by-file)
            with tqdm() as pbar:
                # Path
                root = source.path

                # Extract data : replicate the dataset loading process... (just replace in the future with that??)
                data = dict()
                for attr, filenames in self.filenames.items():
                    for loc, filename in filenames.items():
                        pbar.set_description(f"[{root.name}] Extracting {loc}...")
                        data[loc] = getattr(source, attr)(filename=filename).data()
                        data[loc].attrs["root"] = root
                
                # Extract frame indexes to reference frames
                if self.use_detections:
                    frame_indexes : ndarray = data[self.dkey].frame_indexes(data[self.dkey].data)
                    data[self.fkey].data = data[self.fkey].data[frame_indexes] # ONLY use frames with detections

                # Apply `DataPipe` to data
                pbar.set_description(f"[{root.name}] Processing data...")
                data = self.process(data, pbar=pbar)["data"]

                pbar.set_description(f"[{root.name}] Completed...")

 