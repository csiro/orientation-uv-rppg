import dotenv
dotenv.load_dotenv(".env")

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

import subprocess
from pathlib import Path
from omegaconf import OmegaConf
from mlflow import MlflowClient
from hydra.utils import instantiate
from mlflow.artifacts import download_artifacts
from argparse import ArgumentParser, Namespace, ArgumentTypeError
from typing import *

import warnings
from lightning.fabric.utilities.warnings import PossibleUserWarning
warnings.simplefilter("ignore", category=PossibleUserWarning)


def str2bool(val: Any) -> bool:
    if isinstance(val, bool): return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise ArgumentTypeError('Boolean value expected.')


# Main
_HYDRA_ARTIFACTS = {
    "config": "config.yaml",
    "hydra": "hydra.yaml",
    "overrides": "overrides.yaml"
}
def main(args: Namespace, overrides: List[str]) -> None:
    # TODO: Clean scripts for training and testing to make more readable.
    # Launch client
    client = MlflowClient(tracking_uri=args.tracking_uri)
    run = client.get_run(args.uuid)   

    if not (run.info.status == "FINISHED"):
        log.warning(f"Run {args.uuid} is NOT finished.")
    
    # Load configuration (re-use artifacted configuration)
    '''
    Only component of the configuration we care about is the output directory to copy the model
    graph to, and load this as the model.
    '''
    dir = download_artifacts(
        run_id=args.uuid,
        tracking_uri=args.tracking_uri,
        artifact_path="config/.hydra"
    )
    artifacts = {key: OmegaConf.load(Path(dir).joinpath(val)) for (key, val) in _HYDRA_ARTIFACTS.items()}

     # Restore configuration directory locally
    hydra_dir = Path(artifacts["hydra"].hydra.runtime.output_dir) 
    hydra_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    try:
        checkpoint = download_artifacts(
            run_id=args.uuid,
            tracking_uri=args.tracking_uri,
            artifact_path="model/checkpoints/model.ckpt",
            dst_path=str(hydra_dir.joinpath("model.ckpt"))
        )
    except FileExistsError:
        checkpoint = str(hydra_dir.joinpath("model.ckpt"))
        log.info(f"File already exists, using that file.")

    # 
    os.environ["LIGHTNING_CHECKPOINT_PATH"] = str(checkpoint)
    os.environ["MLFLOW_REFERENCED_RUN_UUID"] = str(args.uuid)
    os.environ["HYDRA_OUTPUT_DIRECTORY"] = str(hydra_dir)

    # Additional Overrides
    overrides += [f"+checkpoint={str(os.environ.get('LIGHTNING_CHECKPOINT_PATH'))}"]

    # Linked
    if not args.linked:
        if "MLFLOW_RUN_UUID" in os.environ:
            del os.environ["MLFLOW_RUN_UUID"] # Launch with new MLFLOW run

    # Launch training
    p = subprocess.Popen(["python", "src/test.py"] + overrides)
    p.wait()


def get_args_parser() -> ArgumentParser:
    import argparse
    parser = argparse.ArgumentParser(
        description="Experiment Launcher [Test]"
    )

    parser.add_argument(
        "--tracking_uri", type=str, required=False,
        default=str(Path(os.environ["PROJECT_ROOT"]).joinpath("artifacts").joinpath("mlruns")),
        help="MLFlow Server Tracking URI"
    )
    parser.add_argument(
        "--uuid", type=str, required=True,
        default=None,
        help="MLFlow Run UUID to use."
    )
    parser.add_argument(
        "--linked", type=str2bool, required=False,
        default=False,
        help="If true will resume LOGGER_EXPERIMENT and append items there."
        # TODO: Will cause issues with overriding params in MLFlow
    )

    return parser


def parse_args(parser: Callable, sargs: Optional[List[str]] = []) -> Tuple[Namespace, List[str]]:
    _parser = parser()
    return _parser.parse_known_args(sargs)
    

def run(sargs: List[str]) -> None:
    args, unparsed = parse_args(get_args_parser, sargs)
    main(args, unparsed)


if __name__ == "__main__":
    run(sys.argv[1:])
