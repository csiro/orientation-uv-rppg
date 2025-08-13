import dotenv
dotenv.load_dotenv(".env")

import os
import sys
import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT_KEY = "PROJECT_ROOT"
PROJECT_ROOT = os.path.abspath(os.environ.get(PROJECT_ROOT_KEY, "."))
if PROJECT_ROOT_KEY not in os.environ:
    log.warning(f"Defaulting {PROJECT_ROOT_KEY} to: {PROJECT_ROOT}")
    os.environ[PROJECT_ROOT_KEY] = PROJECT_ROOT # Set 
sys.path.append(os.environ[PROJECT_ROOT_KEY])

# ----------------------------------------

import hydra
import subprocess
from pathlib import Path
from torch.utils.data import random_split
from src.utils.accelerator import CUDA
from src.utils.hydra import resolve_and_compose_model_processes

from typing import *
from argparse import ArgumentParser, Namespace, ArgumentTypeError


def str2bool(val: Any) -> bool:
    if isinstance(val, bool): return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise ArgumentTypeError('Boolean value expected.')


# Main
def main(args: Namespace, overrides: List[str]) -> None:
    # Compose hydra configuration
    with hydra.initialize(version_base="1.3", config_path=Path("../configs"), job_name="train_cv"):
        cfg = hydra.compose(config_name="train", overrides=overrides)
        cfg = resolve_and_compose_model_processes(cfg)

    # Attributes
    attributes = []
    for file in hydra.utils.instantiate(cfg.datamodule.dataset.source):
        with file as fp:
            model = fp.interface()
            attributes.append(getattr(model, args.attr))
    attributes = list(set(attributes))

    # Groups
    num_splits = len(attributes) if args.folds is None else args.folds
    assert num_splits <= len(attributes), f"Require num"
    fractions = [1/num_splits for _ in range(num_splits)]
    splits : List[Any] = random_split(attributes, fractions)

    # Define overrides for jobs
    cuda = CUDA()
    for idx, split in enumerate(splits):
        # Extract items to use for outer testing split (remainder will be assigned to outer training split)
        if args.holdout.lower() in ["1fold"]:
            # Outer split of (k-1) training and (1) testing folds
            '''
            This is useful for testing the independent performance with respect to one attribute.

            E.g. subject-independent performance.
            '''
            test_items = [attributes[jdx] for jdx in split.indices]

        elif args.holdout.lower() in ["k_1fold"]:
            # Outer split of (1) training and (k-1) testing folds
            '''
            This is useful for testing the generalization performance from one attribute to other attributes.

            E.g. Stationary generalization to other motion conditions
            '''
            train_items = set([attributes[jdx] for jdx in split.indices])
            test_items = list(set(attributes).difference(train_items))

        else:
            raise NotImplementedError(f"Provided hold-out algorithm ({args.holdout}) is not implemented.")
        
        # Copy command parameters
        split_overrides = overrides.copy()
        split_overrides.append(f"+datamodule.dataset.construct.operations.outer_split.test.filters.include_filter.match.{args.attr}={test_items}")

        # Launch training
        if args.launcher.lower() in ["python"]:
            # Partition jobs across the available GPUs to balance the load.
            devices = cuda.balance_devices(idx, len(splits), args.gpus_per, args.vram_per)
            split_overrides.append(f"trainer.devices={devices}")
            log.info(f"[{idx}] Launching cross-validation training for {args.attr} on {test_items}.")

            # Launch training scripts
            p = subprocess.Popen(["python", "scripts/train.py"] + split_overrides, start_new_session=True)

        elif args.launcher.lower() in ["slurm"]:
            raise NotImplementedError(f"Provided launcher ({args.launcher}) is not implemented.")
        
        else:
            raise NotImplementedError(f"Provided launcher ({args.launcher}) is not implemented.")


def get_args_parser() -> ArgumentParser:
    import argparse
    parser = argparse.ArgumentParser(
        description="Cross-validation Experiment Launcher",
        epilog="Please contact Sam Cantrill at sam.cantrill@data61.csiro.au for help."
    )

    # Cross-validation
    group_cv = parser.add_argument_group("k-fold Cross Validation")
    group_cv.add_argument(
        "--attr", type=str, required=True,
        help="Dataset attribute to perform exclusive cross-validation against."
    )
    group_cv.add_argument(
        "--holdout", type=str, required=False,
        default="1fold",
        choices=["1fold","k_1fold"],
        help="Grouping method for the hold-out set. Use 1 fold for independant performance and k-1 folds is for generalization performance."
    )
    group_cv.add_argument(
        "--folds", type=int, required=False,
        default=None,
        help="Number of folds to use for cross-validation. Should be None for stratified populations."
    )

    # Resources
    group_dev = parser.add_argument_group("Launcher & Resources")
    group_dev.add_argument(
        "--launcher", type=str, required=False,
        default="python",
        choices=["python", "slurm"],
        help="Launch training jobs using a specific method."
    )
    group_dev.add_argument(
        "--gpus_per", type=int, required=False,
        default=1,
        help="Devices (GPUs) per training job."
    )
    group_dev.add_argument(
        "--vram_per", type=float, required=False,
        default=10,
        help="Device (GPU) VRAM (GiB) per device per training job."
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
