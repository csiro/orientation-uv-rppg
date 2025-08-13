'''
=== [ Checkpoints ] ===

Checkpoints allow you the resume training from that point, useful in the event where the
training process is interrupted, or you want to resume training with different parameters.

Lightning checkpoints contain a dump of a `LightningModule`'s entire state, saves everything
necessary to restore a model even in distributed training environments. 

Saves the following:
    - precision
    - global_step
    - `LightningModule` state-dict
    - Optimizers state-dict
    - Schedulers state-dict
    - Callbacks state-dict
    - DataModules state-dict
    - Hyperparameters (init arguments) for the model
    - Hyperparameters (init arguments) for the datamodule
    - Loop states

--- Where ---
By default `ModelCheckpoint` will save files into `Trainer.log_dir`, however you can override
the `dirpath` and `filename` if desired, and you can format the `filename` using f-strings e.g.
f"sample_{epoch:02d}-{val_loss:.2f}"


--- Manually ---
You can manually save 


--- Distributed ---
In distributed training, Lightning ensures only ONE checkpoint is saved (rank 0), this requires
no change to the default code.

Use the provided functions e.g. `save_checkpoint` where possible, if overriding then make sure to
utilize the `rank_zero_only` decorator to ensure saving only occurs on the main process.

NOTE: This won't work if different models share different states such as in sharded or deepspeed
based training strategies.


--- Custom Checkpoint ---



--- Asynchronous ---
AsyncCheckpointIO and TorchCheckpointIO


_target_: lightning.pytorch.callbacks.ModelCheckpoint
dirpath: 
    directory to save the model file e.g. /dir/to/save
filename: 
    checkpoint filename e.g. f"{epoch}_{val_loss:.2f}_{other_metric:.2f}"

monitor: 
    quantity to monitor (logs when metric improves)
verbose: 
save_last: 
    saves copy of checkpoint to last.ckpt whenever is saved (restart deterministically)
save_top_k: 
    aves best k models according to `monitor` (0=none, -1=all) and monitor is checked every `every_n_epochs` epochs
mode: 
    min/max of monitor e.g. min for val_loss or max for val_acc
auto_insert_metric_name: 
    will contain metric name
save_weights_only: 
    only save model weights (for deployment)
every_n_train_steps: 
    num training steps between checkpoints None/0 = skip
train_time_interval: 
    num training time between checkpoints (rough estimate to nearest batch)
every_n_epochs: 
    num epochs between checkpoints (mutually excl. with above 2)
save_on_train_epoch_end: 
    whether to run at the end of the train epoch, if false then checks at end of valid
    reommend setting `True` if monitoring a training metric
enable_version_counter:

Generally seem to want two version of the ModelCheckpoints
1. training (time/epoch based) with full information
2. fitted (loss based) with only weights 
3. deployment (metric based) with only weights


--- CSV ---
Implements `experiment`, `log_hyperparams`, `log_metrics`, and `finalize` as required
by all loggers.


--- MLFlow ---

`tracking_uri` defaults to the env `MLFLOW_TRACKING_URI` if not provided 
    oc.select(oc.env:MLFLOW_TRACKING_URI,paths.log_dir)

Creates an `mlflow_client` with the created `tracking_uri`

`experiment` defines the MLFlow object, access this to use MLflow features in your
code.

    Will get the run based on the `run_id` or start a new run if one was not provided,
    and then assign `experiment_id` with the run info id.

    Implements the common `experiment`, `log_hyperparams`, `log_metrics`, and `finalize`
    required by all loggers.

    `run_id` is the current experiment run ID.







'''
from dataclasses import dataclass
from datetime import timedelta

from typing import *


@dataclass
class ModelCheckpoint:
    '''
    You can create multiple `ModelCheckpoint` callbacks if you want to
    checkpoint every N hours, M train batches, or K val epochs.

    If checkpoint's `dirpath` changed the only `best_model_path` will be
    reloaded and a warning will be raised.
    
    '''
    dirpath: Optional[str] = None # defaults to `default_root_dir`
    filename: Optional[str] = None # defaults to {epoch}-{step}

    # Information
    verbose: bool = False
    enable_version_counter: bool = True

    save_last: Optional[bool] = None
    save_top_k: Optional[bool] = None
    save_weights_only: bool = False

    # Checking
    save_on_train_epoch_end: Optional[bool] = True # run check at end of train/val(False)
    
    # Training
    every_n_train_steps: Optional[int] = None
    train_time_interval: Optional[timedelta] = None
    every_n_epochs: Optional[int] = None

    # Metric-based
    monitor: Optional[str] = None # save only for last epoch
    mode: str = {"min", "max"} # min for loss, max for metric
    auto_insert_metric_name: bool = False

    # Attributes
    CHECKPOINT_JOIN_CHAR = "-"
    CHECKPOINT_NAME_LAST = "last"
    FILE_EXTENSION = ".ckpt"
    STARTING_VERSION = 1
