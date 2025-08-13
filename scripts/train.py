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
from functools import partial
from mlflow.artifacts import download_artifacts
from argparse import ArgumentParser, Namespace, ArgumentTypeError
from typing import *


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
    # Launch client
    client = MlflowClient(tracking_uri=args.tracking_uri)

    # Extract info from referenced run
    if args.uuid is not None:

        # Retrieve MLFlow Run
        run = client.get_run(args.uuid)

        # 
        os.environ["MLFLOW_REFERENCED_RUN_UUID"] = str(args.uuid)

        # Don't resume if overrides were provided
        if args.resume and (len(overrides) > 0): args.resume = False

        # Don't resume if run has finished
        if args.resume and (run.info.status == "FINISHED"): args.resume = False

        # Load configuration (re-use artifacted configuration)
        dir = download_artifacts(
            run_id=args.uuid,
            tracking_uri=args.tracking_uri,
            artifact_path="config/.hydra"
        )
        artifacts = {key: OmegaConf.load(Path(dir).joinpath(val)) for (key, val) in _HYDRA_ARTIFACTS.items()}

        # Restore configuration directory locally
        hydra_dir = Path(artifacts["hydra"].hydra.runtime.output_dir)
        hydra_dir.mkdir(parents=True, exist_ok=True)

        for subdir in ["configs", ".hydra"]:
            download_artifacts(
                run_id=args.uuid, 
                tracking_uri=args.tracking_uri, 
                artifact_path=f"config/{subdir}", 
                dst_path=str(hydra_dir)
            )

        # Set : hydra dir + hydra config + updated overrides
        os.environ["HYDRA_CONFIG_PATH"] = str(hydra_dir.joinpath("config/configs"))
        os.environ["HYDRA_CONFIG_NAME"] = str(artifacts["hydra"].hydra.job.config_name)
        overrides = artifacts["overrides"] + overrides
    

        if (args.resume) or (args.checkpoint is not None):

            # Resume
            if args.resume:
                os.environ["MLFLOW_RUN_UUID"] = str(args.uuid)
                os.environ["HYDRA_OUTPUT_DIRECTORY"] = str(hydra_dir)
            
            # Fallback to default checkpoint value if un-defined
            if args.checkpoint is None:
                args.checkpoint = "**/*.ckpt"
            
            # Restore checkpoint
            dir = download_artifacts(run_id=args.uuid, tracking_uri=args.tracking_uri, artifact_path="model/checkpoints")
            checkpoints = list(sorted(Path(dir).glob(args.checkpoint), key = lambda file: os.path.getctime(file)))
            checkpoint = checkpoints[-1] if args.last else checkpoints[0]

            # 
            os.environ["LIGHTNING_CHECKPOINT_PATH"] = str(checkpoint)

    # Launch training
    p = subprocess.Popen(["python", "src/train.py"] + overrides)
    p.wait()


def get_args_parser() -> ArgumentParser:
    import argparse
    parser = argparse.ArgumentParser(
        description="Experiment Launcher [Train]"
    )

    parser.add_argument(
        "--tracking_uri", type=str, required=False,
        default=str(Path(os.environ["PROJECT_ROOT"]).joinpath("artifacts").joinpath("mlruns")),
        help="MLFlow Server Tracking URI"
    )
    parser.add_argument(
        "--uuid", type=str, required=False,
        default=None,
        help="MLFlow Run UUID to use."
    )
    parser.add_argument(
        "--resume", type=str2bool, required=False,
        default=True,
        help="Resume MLFlow run."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=False,
        default=None,
        help="Checkpoint regular expression to glob."
    )
    parser.add_argument(
        "--last", type=str2bool, required=False,
        default=True,
        help="Use last configuration in globbed items."
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
