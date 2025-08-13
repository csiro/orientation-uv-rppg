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

import yaml
import numpy as np
from pathlib import Path
from argparse import ArgumentParser, Namespace, ArgumentTypeError
from typing import *

def main(args: Namespace, overrides: List[str]) -> None:
    # Resolve tracking URI
    args.tracking_uri = Path(args.tracking_uri).resolve().absolute()
    
    # Experiment ID
    experiments = {}
    for path in args.tracking_uri.glob(f"*/meta.yaml"):
        with open(path, "r") as fp:
            data = yaml.safe_load(fp)
            experiments[data["name"]] = path.parent
            # print(f"{data['name']}: {path.parent.stem}")

    if args.experiment is None:
        raise ValueError(f"Experiment name not provided. Use one of: {list(experiments.keys())}")
    
    # Runs
    runs = {p.parent.stem:p.parent for p in experiments[args.experiment].glob(f"*/meta.yaml")}

    # --- [ Parameters ] ---
    if args.params is not None:
        print(" --- [ PARAMETERS ] ---")
        for run_uuid in args.uuid:
            for p in runs[run_uuid].glob("params/*"):
                if any(param in p.name for param in args.params):
                    with open(p, "r") as fp:
                        val = fp.readline()
                        print(f"[{run_uuid}] {p.name}: {val}")
        print("")

    # --- [ Metrics ] ---
    # Access metrics
    metrics = {}
    for run_uuid in args.uuid:
        for p in runs[run_uuid].glob("metrics/*"):
            metrics[f"{run_uuid}/{p.name}"] = p

    if args.metrics is not None:
        # Load metrics
        results = {}
        for key, path in metrics.items():
            for metric in args.metrics:
                if key.split("/")[-1] == metric:
                    with open(path, "r") as fp:
                        data = [[float(v) for v in line[:-1].split(" ")] for line in fp.readlines()]
                    results[key] = np.array(data)

        # Print metrics
        print("--- [ METRICS ] ---")
        for run, data in results.items():
            uuid, metric = tuple(run.split("/"))
            min_idx = np.argmin(data[:,1])
            min_val = data[min_idx,1]
            min_step = data[min_idx,2]
            print(f"[{uuid} | {metric}] min {min_val:.5f} @ {min_step} (index={min_idx})")
        print("")
    else:
        metrics = (set([k.split("/")[-1] for k in metrics.keys()]))
        raise ValueError(f"Metrics not provided. Use one/any of [{metrics}]")


    # --- [ Step ] ---
    if args.step is not None:
        print("--- [ RESULTS @ STEP ] ---")
        for step in args.step:
            for run, data in results.items():
                # Info
                uuid, metric = tuple(run.split("/"))

                # Minimum
                min_idx = np.argmin(data[:,1])
                min_val = data[min_idx,1]

                # Step
                use_idx = np.argmin(np.abs(data[:,2] - step))
                nearest_step = data[use_idx,2]
                use_val = data[use_idx,1]

                # Results
                print(f"[{metric} | {uuid}] min: {min_val:.5f} @ {min_idx} | use ({step}) {use_val:.5f} @ {nearest_step}")
        print("")


def get_args_parser() -> ArgumentParser:
    import argparse
    parser = argparse.ArgumentParser(
        description="Experiment Results Summary"
    )

    parser.add_argument(
        "--tracking_uri", type=str, required=False,
        default=str(Path(os.environ["PROJECT_ROOT"]).joinpath("artifacts").joinpath("mlruns")),
        help="MLFlow Server Tracking URI"
    )
    parser.add_argument(
        "--uuid", type=str, required=False,
        default=None, 
        action="extend", nargs="+",
        help="MLFlow Run UUID to use."
    )

    parser.add_argument(
        "--experiment", type=str, required=False,
        default=None,
        help="MLFlow experiment name to use."
    )

    parser.add_argument(
        "--params", type=str, required=False,
        default=None,
        action="extend", nargs="+",
        help="MLFlow params to use."
    )
    parser.add_argument(
        "--metrics", type=str, required=False,
        default=None,
        action="extend", nargs="+",
        help="MLFlow metrics to use."
    )
    parser.add_argument(
        "--step", type=int, required=False,
        default=None,
        action="extend", nargs="+",
        help="Global optimizer step to obtain results for."
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