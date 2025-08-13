import os
import sys
import logging
log = logging.getLogger(__name__)

# Add project to system path
PROJECT_ROOT_KEY = "PROJECT_ROOT"
PROJECT_ROOT = os.path.abspath(os.environ.get(PROJECT_ROOT_KEY, "."))

if PROJECT_ROOT_KEY not in os.environ:
    log.warning(f"Defaulting {PROJECT_ROOT_KEY} to: {PROJECT_ROOT}")
    os.environ[PROJECT_ROOT_KEY] = PROJECT_ROOT # Set 

sys.path.append(os.environ[PROJECT_ROOT_KEY])

# ----------------------------------------

import re
import torch
import hydra
from src import * # Resolvers
from pathlib import Path
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from src.utils.hydra import resolve_and_compose_model_processes
from lightning.pytorch import seed_everything
from src.utils.profilers import instantiate_profiler
from src.utils.callbacks import instantiate_callbacks

from typing import *
from omegaconf import DictConfig
from src.utils.loggers import Loggers
from lightning import Trainer, Callback
from lightning.pytorch.profilers import Profiler


# Main
_HYDRA_KWARGS = {
    "version_base": os.environ.get("HYDRA_VERSION_BASE", "1.3"),
    "config_path": os.environ.get("HYDRA_CONFIG_PATH", str(Path(PROJECT_ROOT).joinpath("configs"))),
    "config_name": os.environ.get("HYDRA_CONFIG_NAME", "test.yaml")
}
@hydra.main(**_HYDRA_KWARGS)
def main(cfg: DictConfig) -> str:
    # Parse configuration
    cfg = resolve_and_compose_model_processes(cfg)
    
    # Seed
    if "seed" in cfg: 
        seed_everything(cfg.seed, workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Loggers
    loggers : Loggers = Loggers(cfg.loggers)

    # Callbacks
    profiler : Profiler = instantiate_profiler(cfg.profiler)
    callbacks : List[Callback] = instantiate_callbacks(cfg.callbacks)
    
    # Trainer
    trainer : Trainer = instantiate(
        cfg.trainer, 
        logger=loggers.values(), 
        callbacks=list(callbacks.values()), 
        profiler=profiler
    )
    os.environ["NUM_DEVICES"] = f"{trainer.num_devices}"

    # DataModule & Model
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    os.environ["TRAIN_SAMPLES"] = str(datamodule.information()["length_train"]) # Not used
    model = hydra.utils.instantiate(cfg.model, datamodule=datamodule)

    # Logging
    envinfo = ["MLFLOW*", "HYDRA*", "LIGHTNING*"]
    envinfo = {f"environment_information/{k}":os.environ[k] for k in os.environ.keys() if any([re.match(q,k) is not None for q in envinfo])}
    cfginfo = OmegaConf.to_container(cfg, resolve=True)
    datainfo = {f"datamodule_information/{key}":val for key,val in datamodule.information().items()} # Dataset : training
    modelinfo = {f"model_information/{key}":val for key,val in model.information().items()}

    # Log hydra configuration
    for logger in loggers.values():
        logger.log_hyperparams({**cfginfo, **envinfo, **datainfo, **modelinfo})

    # Artifact hydra configuration to MLFlow (on new run initialization)
    loggers.get().push_artifacts(_HYDRA_KWARGS["config_path"], "config/configs")
    loggers.get().push_artifacts(Path(HydraConfig.get().runtime.output_dir).joinpath(".hydra"), "config/.hydra")

    # Testing
    trainer.test(model, datamodule, ckpt_path=cfg.checkpoint if "checkpoint" in cfg else os.environ.get("LIGHTNING_CHECKPOINT_PATH", None))

    # Finalize
    loggers.finish()


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
