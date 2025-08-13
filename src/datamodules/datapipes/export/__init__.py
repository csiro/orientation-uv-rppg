import h5py
from pathlib import Path
from src.datamodules.datamodels import DataModel
from src.datamodules.datamodels import DatasetSampleKeys
from src.datamodules.datapipes import DataOperation

from typing import *
from torch import Tensor
from numpy import ndarray


class DatasetSampleWriter:
    """

    """
    def __init__(self,
        path: str, 
        model_name: Optional[str] = "not_available",
        model_uuid: Optional[str] = "not_available",

        write_inputs: Optional[bool] = False,
        write_targets: Optional[bool] = False,
        write_sources: Optional[bool] = False,
        write_outputs: Optional[bool] = False,
        write_predictions: Optional[bool] = False,
        write_losses: Optional[bool] = False,
        write_metrics: Optional[bool] = False,
        
        write_inputs_data: Optional[bool] = False,
        write_targets_data: Optional[bool] = False,
        write_sources_data: Optional[bool] = False,
        write_outputs_data: Optional[bool] = False,
        write_predictions_data: Optional[bool] = False,
        write_losses_data: Optional[bool] = False,
        write_metrics_data: Optional[bool] = False,

        export_batch: Optional[bool] = True,
        *args, **kwargs
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.model_uuid = model_uuid

        self.write_inputs = write_inputs
        self.write_targets = write_targets
        self.write_sources = write_sources
        self.write_outputs = write_outputs
        self.write_predictions = write_predictions
        self.write_losses = write_losses
        self.write_metrics = write_metrics

        self.write_inputs_data = write_inputs_data
        self.write_targets_data = write_targets_data
        self.write_sources_data = write_sources_data
        self.write_outputs_data = write_outputs_data
        self.write_predictions_data = write_predictions_data
        self.write_losses_data = write_losses_data
        self.write_metrics_data = write_metrics_data

        self.export_batch = export_batch

    def __call__(self,
        inputs: Optional[Dict[str, DataModel]] = None,
        targets: Optional[Dict[str, DataModel]] = None,
        sources: Optional[Dict[str, DataModel]] = None,
        outputs: Optional[Dict[str, DataModel]] = None,
        predictions: Optional[Dict[str, DataModel]] = None,
        losses: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        use_sample_key: Optional[str] = None,
        *args, **kwargs
    ) -> None:
        """

        TODO: Overhaul implementation to more cleaning handle writing batched and entry `DataModel`.

        Structure of items to write is typically:
            DataModel
        """
        # update source with additional data (relevant for identifying post evaluation)
        for key, val in sources.items():
            if self.export_batch: # if exporting non-batch (e.g. not testing then leave as-is)
                n_keys = len(val.attrs["key"])
                if "model_name" not in val.attrs: val.attrs["model_name"] = [self.model_name] * n_keys
                if "model_uuid" not in val.attrs: val.attrs["model_uuid"] = [self.model_uuid] * n_keys

        # write to file (dump)
        with h5py.File(self.path, "a") as fp:
            srcs_key = sources["sample"].attrs["key"] if self.export_batch else list(range(1))
            for idx, sample_key in enumerate(srcs_key): # per source
                # clear pre-existing group (i.e. re-run testing with exact same params)
                sample_key = sample_key if use_sample_key is None else use_sample_key # use a pre-defined key
                sample_key = str(sample_key)
                if sample_key in fp: del fp[sample_key]

                # create sample groups
                sample_group = fp.create_group(sample_key.replace("/", "_"))

                # write source
                if self.write_sources:
                    if sources is not None:
                        # create entry & dump `DataModel`s
                        entry_group = sample_group.create_group(DatasetSampleKeys.SOURCE.value)
                        for key, val in sources.items():
                            val.dump(entry_group, key, self.write_sources_data, idx if self.export_batch else None)

                # write inputs
                if self.write_inputs:
                    if inputs is not None:
                        # create entry & dump `DataModel`s
                        entry_group = sample_group.create_group(DatasetSampleKeys.INPUTS.value)
                        for key, val in inputs.items():
                            val.dump(entry_group, key, self.write_inputs_data, idx if self.export_batch else None)
                
                # write targets
                if self.write_targets:
                    if targets is not None:
                        # create entry & dump `DataModel`s
                        entry_group = sample_group.create_group(DatasetSampleKeys.TARGETS.value)
                        for key, val in targets.items():
                            val.dump(entry_group, key, self.write_targets_data, idx if self.export_batch else None)

                # write outputs (model training outputs)
                if self.write_outputs:
                    if outputs is not None:
                        # create entry & dump `DataModel`s
                        entry_group = sample_group.create_group(DatasetSampleKeys.OUTPUTS.value)
                        for key, val in outputs.items():
                            val.dump(entry_group, key, self.write_outputs_data, idx if self.export_batch else None)

                # write predictions (model inference outputs)
                if self.write_predictions:
                    if predictions is not None:
                        # create entry & dump `DataModel`s
                        entry_group = sample_group.create_group(DatasetSampleKeys.PREDICTIONS.value)
                        for key, val in predictions.items():
                            val.dump(entry_group, key, self.write_predictions_data, idx if self.export_batch else None)

                # construct losses `DataModel` and write losses
                if self.write_losses:
                    if losses is not None:
                        # check
                        assert idx == 0, f"Testing requires a batch size of one currently due to metric calculation being aggregated."

                        # create entry
                        entry_group = sample_group.create_group(DatasetSampleKeys.LOSSES.value)
                        for key, val in losses.items():
                            assert idx == 0, f"Can only export non accumulated losses/metrics TODO"
                            # construct `DataModel` for losses
                            model = DataModel(data=val, attrs={"name": key})

                            # dump `DataModel`
                            model.dump(entry_group, key.replace("/","_"), self.write_losses_data, idx if self.export_batch else None)

                # construct metrics `DataModel` and write metrics
                if self.write_metrics:
                    if metrics is not None:
                        # check
                        assert idx == 0, f"Testing requires a batch size of one currently due to metric calculation being aggregated."

                        # create entry
                        entry_group = sample_group.create_group(DatasetSampleKeys.METRICS.value)
                        for key, val in metrics.items():
                            assert idx == 0, f"Can only export non accumulated losses/metrics TODO"
                            # construct `DataModel` for losses
                            model = DataModel(data=val, attrs={"name": key})

                            # dump `DataModel`
                            model.dump(entry_group, key.replace("/","_"), self.write_metrics_data, idx if self.export_batch else None)


class DatasetSamplesWriter(DataOperation):
    def __init__(self, keys: Dict[str,str], writer: Callable, *args, **kwargs) -> None:
        super(DatasetSamplesWriter, self).__init__(*args, **kwargs)
        self.writer = writer
        self.keys = keys

    def __call__(self, videos: Dict[str, Dict[str, Any]]) -> None:
        for idx, (name, vals) in enumerate(videos.items()):
            kwargs = {}
            for dst_key, src_key in self.keys.items():
                _src_key = src_key.split("/")[-1]
                kwargs[dst_key] = {_src_key: vals[src_key]}
            kwargs["use_sample_key"] = name
            self.writer(**kwargs)
            # if idx >= 0: break
        return str(self.writer.path)
