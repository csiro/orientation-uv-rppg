import torch
import numpy as np
from tqdm import tqdm
from abc import abstractmethod
from src.datamodules.datamodels import DataModel
from src.datamodules.datapipes import DataOperation
from src.datamodules.datamodels import DatasetSampleKeys
from src.datamodules.datamodels.timeseries import TimeseriesModel

from typing import *
from numpy import ndarray
from torch import Tensor


class GroupVideoDatasetSamples(DataOperation):
    def __init__(self, *args, **kwargs) -> None:
        super(GroupVideoDatasetSamples, self).__init__(*args, **kwargs)

    def __call__(self, files: Iterable) -> Iterable:
        grouped_samples = {}

        with tqdm() as pbar:
            for file in files:
                # iterate over samples in file
                for idx, sample in enumerate(file):

                    # get video name to use
                    sample_video_name = sample["source/sample"].attrs["video_name"]

                    # assign to video
                    if sample_video_name in grouped_samples:
                        grouped_samples[sample_video_name].append(sample)
                    else:
                        grouped_samples[sample_video_name] = [sample]

                    pbar.update(1)
            
        return grouped_samples


class CombineVideoDatasetSamples(DataOperation):
    def __init__(self, *args, **kwargs) -> None:
        super(CombineVideoDatasetSamples, self).__init__(*args, **kwargs) 

    def __call__(self, data: Dict[str, List[Dict[str, Any]]]) -> None:
        videos = {}

        for video_name, video_samples in tqdm(data.items()):
            # sort by index or key
            indexes = list(sorted(sample["source/sample"].attrs["index"] for sample in video_samples))
            video_samples = {sample["source/sample"].attrs["index"]: sample for sample in video_samples}

            # define a new sample for the video
            combined_sample = {}

            # iterate over sorted samples
            for index in indexes:
                # current sample
                sample = video_samples[index]

                # entries in sample
                for key, val in sample.items():
                    # add key to combined sample
                    if key not in combined_sample:
                        combined_sample[key] = DataModel(data=[], attrs={})

                    # accumulate data
                    combined_sample[key].data.append(val.data)

                    # accumulate attrs
                    for k, v in val.attrs.items():
                        if k in combined_sample[key].attrs:
                            combined_sample[key].attrs[k].append(v)
                        else:
                            combined_sample[key].attrs[k] = [v]

            # stack combined data (but not attrs?)
            for key, val in combined_sample.items(): 
                # concantenate data
                if len(val.data) > 0:
                    if isinstance(val.data[0], ndarray):
                        val.data = np.concatenate(val.data, axis=0)
                    else:
                        val.data = np.array(val.data)

                # stack attrs
                for k, v in val.attrs.items():
                    if isinstance(v, list):
                        if len(v) > 0:
                            if isinstance(v[0], ndarray):
                                val.attrs[k] = np.concatenate(v, axis=0)
                            else:
                                val.attrs[k] = list(set(v))
                                if len(val.attrs[k]) == 1:
                                    val.attrs[k] = val.attrs[k][0] # un-list if single val
                    # print(k, val.attrs[k])

            # add to combined video
            videos[video_name] = combined_sample
        
        return videos
                    


class ProcessVideoDatasetSamples(DataOperation):
    def __init__(self, process: Callable, *args, **kwargs) -> None:
        super(ProcessVideoDatasetSamples, self).__init__(*args, **kwargs)
        self.process = process

    def __call__(self, videos: Dict[str, Dict[str, Any]]) -> None:
        for key, val in tqdm(videos.items()):
            self.process(val)
            # if idx >= 0: break
        return videos


class MetricsVideoDatasetSamples(DataOperation):
    def __init__(self, target_key: str, output_key: str, metrics: Callable, *args, **kwargs) -> None:
        super(MetricsVideoDatasetSamples, self).__init__(*args, **kwargs)
        self.output_key = output_key
        self.target_key = target_key
        self.metrics = metrics

    def __call__(self, videos: Dict[str, Dict[str, Any]]) -> None:
        metrics = []
        for key, val in videos.items():
            # predictions
            output_group_key, output_entry_key = tuple(self.output_key.split("/"))
            outputs = {output_entry_key: val[self.output_key]}
            outputs[output_entry_key].data = outputs[output_entry_key].data.unsqueeze(0)

            # targets
            target_group_key, target_entry_key = tuple(self.target_key.split("/"))
            targets = {target_entry_key: val[self.target_key]}
            targets[target_entry_key].data = targets[target_entry_key].data.unsqueeze(0)

            # compute metrics for current batch
            m = self.metrics(key, outputs, targets, None, False) # results from batch (1)
            metrics.append(m) # accumulate

        return metrics


class GroupMetricsAcrossVideos(DataOperation):
    def __init__(self, *args, **kwargs) -> None:
        super(GroupMetricsAcrossVideos, self).__init__(*args, **kwargs)

    def __call__(self, metrics: List[Dict[str, Tensor]]) -> None:
        grouped_metrics = {}
        for vals in metrics: # [{}, {}, {}]
            for key, val in vals.items():
                # split name
                video_name, _, metric_name = tuple(key.split("/"))

                # store val
                if metric_name in grouped_metrics:
                    if video_name in grouped_metrics[metric_name]:
                        grouped_metrics[metric_name][video_name].append(val.item())
                    else:
                        grouped_metrics[metric_name][video_name] = [val.item()]
                else:
                    grouped_metrics[metric_name] = {video_name: [val.item()]}
        return grouped_metrics

import json
from pathlib import Path

class ExportVideoMetrics(DataOperation):
    def __init__(self, filepath: str, *args, **kwargs) -> None:
        super(ExportVideoMetrics, self).__init__(*args, **kwargs)
        filepath = Path(filepath).absolute().resolve()
        self.filepath = filepath.parent.joinpath(f"{filepath.stem}.json")

    def __call__(self, metrics: List[Dict[str, Tensor]]) -> None:
        with open(self.filepath, "w") as fp:
            json.dump(metrics, fp)
