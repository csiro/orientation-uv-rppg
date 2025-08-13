import dotenv
dotenv.load_dotenv(".env")

import os
import sys
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Root Directory
PROJECT_ROOT_KEY = "PROJECT_ROOT"
PROJECT_ROOT = os.path.abspath(os.environ.get(PROJECT_ROOT_KEY, "."))
if PROJECT_ROOT_KEY not in os.environ:
    log.warning(f"Defaulting {PROJECT_ROOT_KEY} to: {PROJECT_ROOT}")
    os.environ[PROJECT_ROOT_KEY] = PROJECT_ROOT # Set 
sys.path.append(os.environ[PROJECT_ROOT_KEY])

# Submodule Directories
for submodule in Path(PROJECT_ROOT).joinpath("submodules").glob("*"):
    sys.path.append(str(submodule))

# ------------------------------------------

import os
import sys
import hydra
from omegaconf import OmegaConf, DictConfig

from src import *

from typing import *

_HYDRA_KWARGS = {
    "version_base": os.environ.get("HYDRA_VERSION_BASE", "1.3"),
    "config_path": os.environ.get("HYDRA_CONFIG_PATH", str(Path(os.environ["PROJECT_ROOT"]).joinpath("configs"))),
    "config_name": os.environ.get("HYDRA_CONFIG_NAME", None)
}
@hydra.main(**_HYDRA_KWARGS)
def main(cfg: DictConfig) -> None:
    """_summary_

    Args:
        cfg (DictConfig): _description_
    """
    inputs = hydra.utils.instantiate(cfg.inputs)
    processes = hydra.utils.instantiate(cfg.processes)
    outputs = processes(inputs)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
