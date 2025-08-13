import logging
from omegaconf import OmegaConf, DictConfig
from hydra.types import TaskFunction
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback

from typing import *

log = logging.getLogger(__name__)


class Callback:
    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        """
        Called in RUN mode before job/application code starts. `config` is composed with overrides.
        Some `hydra.runtime` configs are not populated yet.
        See hydra.core.utils.run_job for more info.
        """
        ...

    def on_run_end(self, config: DictConfig, **kwargs: Any) -> None:
        """
        Called in RUN mode after job/application code returns.
        """
        ...

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        """
        Called in MULTIRUN mode before any job starts.
        When using a launcher, this will be executed on local machine before any Sweeper/Launcher is initialized.
        """
        ...

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        """
        Called in MULTIRUN mode after all jobs returns.
        When using a launcher, this will be executed on local machine.
        """
        ...

    def on_job_start(self, config: DictConfig, *, task_function: TaskFunction, **kwargs: Any) -> None:
        """
        Called in both RUN and MULTIRUN modes, once for each Hydra job (before running application code).
        This is called from within `hydra.core.utils.run_job`. In the case of remote launching, this will be executed
        on the remote server along with your application code. The `task_function` argument is the function
        decorated with `@hydra.main`.
        """
        ...

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
        """
        Called in both RUN and MULTIRUN modes, once for each Hydra job (after running
        application code).
        This is called from within `hydra.core.utils.run_job`. In the case of remote launching, this will be executed
        on the remote server after your application code.

        `job_return` contains info that could be useful for logging or post-processing.
        See hydra.core.utils.JobReturn for more.
        """
        ...


class MLFlow(Callback):
    """_summary_

    on_run_<start/end>: 
        Called in RUN mode BEFORE/AFTER job starts. 
        Not everything will be populated BEFORE.
    on_multirun_<start/end>: 
        Called in MULTIRUN mode BEFORE/AFTER launcher/sweeper created.
        Executed on local machine.
    on_job_<start/end>: 
        Called in both RUN and MULTIRUN per Hydra job.
        Will be executed on remote server with code.

    """
    def __init__(self, *args, **kwargs) -> None:
        super(MLFlow, self).__init__(*args, **kwargs)

    def on_job_start(self, config: DictConfig, *, task_function: TaskFunction, **kwargs: Any) -> None:
        
        # log.info("uuid" in config)
        # log.info("resume" in config)
        # log.info("checkpoint" in config)
        # log.info(config.hydra.overrides.task)


        '''
        dict_keys(['hydra', 'paths', 'seed', 'loggers', 'callbacks', 'profiler', 'trainer', 'datamodule', 'model'])
        dict_keys(['run', 'sweep', 'launcher', 'sweeper', 'help', 'hydra_help', 'hydra_logging', 'job_logging', 'env', 'mode', 'searchpath', 'callbacks', 'output_subdir', 'overrides', 'job', 'runtime', 'verbose'])
        '''

        # cfg.hydra.overrides.task
        
        # if "uuid" in config:

        #     log.info("set config")

        #     if ("resume" in config) or ("checkpoint" in config):

        #         if "checkpoint" not in config:
                    
        #             log.info("use default checkpoint")

        #         log.info(f"Set checkpoint")

        # config.uuid = "NEW VALUE"

        # config.hydra.overrides.task = config.hydra.overrides.task + ["+KEY=VAL"]

        # log.info(config)

        # log.info("MLFLOW PRE CALLBACK")

    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
        # TODO: Save log files to mlflow for visualization
        pass


def resolve_and_compose_model_processes(cfg: DictConfig) -> DictConfig:
    """ Resolves pre-processing `DataPipe` procedures defined by a specific `Network` and
    composes them into the `Dataset` `DataPipe` processes. 

    Essentially allows model-defined processing of the dataset, allowing models to be swapped
    out automatically without having to manually adjust the dataset processing.

    Args:
        cfg (DictConfig): _description_

    Returns:
        DictConfig: _description_
    """
    # Ensure OmegaConf is in struct mode
    OmegaConf.set_struct(cfg, False)

    # Resolve configuration items
    merge = []
    items = cfg.model.process.resolve.items()

    # Iteratively resolve and update `Network` processes to merge with `Dataset` processes
    for key, val in items:
        # Register resolver for current set
        OmegaConf.register_new_resolver("assign", lambda x: val[x], replace=True)

        # Resolve keys in the processes
        OmegaConf.resolve(cfg.model.process[key])

        # Clear resolver
        OmegaConf.clear_resolver("assign")

        # Update list of processes to merge into the `Dataset` processes (inputs and targets)
        if key in ["inputs", "targets"]: 
            merge.append(cfg.model.process[key])

    # Merge resolved configurations into the `Dataset` processes
    cfg.datamodule.dataset.process = OmegaConf.merge(cfg.datamodule.dataset.process, *merge)

    # Re-enable struct-mode for the configuration
    OmegaConf.set_struct(cfg, True)

    return cfg
