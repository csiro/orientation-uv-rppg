'''
=== [ Logger ] ===

Supports a number of loggers such as `CSVLogger`, `MLFlowLogger`, and `WandbLogger`, you
can also implement custom logging strategies.


- Loggers -
You can pass a list of loggers to the training to utilize multiple loggers with
`logger=[logger1, logger2]`.


- Logging -
...


- Logging from LightningModule -
`LightningModule` offers automatic logging functionality for scalars, or manual logging
for other items, use the `self.log` or `self.log_dict` method anywhere in the module.

Depending on where `log` is call, lightning will determine logging mode for you, you can
override this by manually setting the `log` parameters `on_step`, `on_epoch`, `prog_bar`,
and `logger=True`.

During training will attempt to calculate batch size with `on_epoch=True` to acculuate
metrics, can provide `batch_size=...` to avoid any issues with automatically determining
the size.

NOTE: Setting `on_epoch=True` will cache logged values across epoch and perform reduction
in `on_train_epoch_end`. Recommend using `torchmetrics` when working with custom reduction
function.

NOTE: Due to overhead associated with logging, it may slow down training performance if
logging every step, you can use `log_every_n_steps` to change this behaviour.


- Artifacts - 
When logging non-scalar artifacts such as histograms, text, images, you may need
to directly use the logger object. 


- Hyperparameters -
When training a model, it's useful to know what hyperparameters went into the model, when
lightning creates a checkpoint it stores the `hyperparameters` to allow resuming.

NOTE: If you want to override the default logging behaviour you can use the `on_train_start`
hook.


- Custom Logger -
...


- Logging Frequency -
Indivudual logger implementations control their own flushing frequency, `CSVLogger` for
example allows you to tset the flag `flush_logs_every_n_steps`.


- Progress Bar - 
You can add metrics to the progress bar by setting `prog_bar=True`

Default includes the training loss and version number if using a logger, can override this
with the `get_metrics` hook in your logger.


- Console Logging -
...


- Logging

'''
'''
        `Metric` logging in Lightning occurs through `self.log` and `self.log_dict` methods. Both
        support logging of scalar tensors.

        When `Metric` or `MetricCollection` objects are logged inside a `LightningModule` using the
        `self.log` method, the metrics will be logged based on the `on_step` and `on_epoch` flags
        present. If `on_epoch` is True, then logger automatically logs at the end of the epoch.

        When you call `self.log` with a key:val pair, it will make these metrics available to
        other callback functions such as `EarlyStopping` or any monitoring function.
        
        When using `on_epoch=True` should generally set `sync_dist=True` to accumulate metrics
        across devices, however may incur significant communication overhead.

        Decided to utilize `on_epoch=False` as `EarlyStopping` only makes a decision based on
        the validation loss, not the relative divergence.

        NOTE: `sync_dist` will have no impact on logged `Metric` objects, their contain their
        own internal distributed synchronization logic, stick to the default behaviour.

        NOTE: If you want to calculate epoch-level metrics use `log` for automatic accumulation
        across batches in an epoch, `log` also handles reduction of metrics across devices.

        NOTE: Recommended to directly log the `Metric` objects to allow flags in `self.log` to
        control when items are logged.

        TODO: If we directly log the `Metric` is the synchronization in distributed model handled
        automatically?
        '''

# import re
# from lightning.fabric.utilities.logger import _flatten_dict
# from typing import *

# def format_configuration_params(config: Dict, modules : Optional[List[str]] = []) -> Dict[str,Any]:
#     params = {}
#     for k,v in _flatten_dict(config).items():
#         if len(modules):
#             for q in modules:
#                 if re.compile(q).match(k) is not None:
#                     if "_partial_" not in k:
#                         k = k.replace("/","_")
#                         params[k] = v
#         else:
#             if "_partial_" not in k:
#                 k = k.replace("/","_")
#                 params[k] = v
#     return params

from typing import *
from argparse import Namespace

import mlflow
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
from lightning.fabric.utilities.logger import _flatten_dict


class MLFlowLoggerInterface(MLFlowLogger):
    """_summary_

    Args:
        MLFlowLogger (_type_): _description_
    """
    def __init__(self, *args, **kwargs) -> None:
        super(MLFlowLoggerInterface, self).__init__(*args, **kwargs)

    def _format(self, params: Dict[str, Any]) -> None:
        _params = {}
        for key, val in _flatten_dict(dict(params)).items():
            key = key.replace("/", "_")
            _params[key] = val
        return _params

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        super().log_hyperparams(self._format(params))

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        super().log_metrics(self._format(metrics), step)

    def push_artifacts(self, input: str, output: str) -> None:
        self.experiment.log_artifacts(
            self.run_id,
            str(input),
            str(output)
        )

    def pull_artifacts(self, path: str) -> Any:
        return mlflow.artifacts.download_artifacts(
            run_id=self.run_id, 
            tracking_uri=self.experiment.tracking_uri,
            artifact_path=str(path)
        )
    

class WandbLoggerInterface(WandbLogger):
    """_summary_

    Args:
        WandbLogger (_type_): _description_
    """
    # TODO: Logging ConfusionMatrix or more complex values or artifacts

    def __init__(self, *args, **kwargs) -> None:
        super(WandbLoggerInterface, self).__init__(*args, **kwargs)
        self.LOGGER_JOIN_CHAR = "/"

import os
from src.utils import instantiate_items
from omegaconf import DictConfig

def instantiate_loggers(cfg: DictConfig, keys: Optional[List[str]] = ["mlflow"]) -> Dict[str, Any]:
    def init_uuid(key: str, mlflow_logger: WandbLoggerInterface) -> None:
        if key in keys:
            os.environ["MLFLOW_RUN_UUID"] = str(mlflow_logger.run_id)
    return instantiate_items(cfg, keys, init_uuid)


from omegaconf import DictConfig

class Loggers:
    """_summary_

    Interface class for handling multple loggers.

    """
    def __init__(self, cfg: DictConfig, primary: Optional[str] = "mlflow") -> None:
        super(Loggers, self).__init__()
        self.primary = primary
        self.loggers = instantiate_loggers(cfg, keys=[self.primary])

    def values(self) -> List[Logger]:
        return list(self.loggers.values())

    def finish(self) -> None:
        for logger in self.values():
            if hasattr(logger.experiment, "finish"):
                logger.experiment.finish()
    
    def get(self, key: Optional[str] = None) -> Any:
        return self.loggers[self.primary if key is None else self.loggers[key]]