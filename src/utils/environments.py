'''
=== ENVIRONMENTS ===

--- Lightning ---
Default environment for a single node or non-managed free cluster, can operate in two modes

1. User only launches the main process `python train.py ...` with NO environment variables
set
2. User launches all the processes manually or with utilities `torch.distributed.launch` with
the appropriate environment variables set.


--- MPI ---
...



--- SLURM ---
Cluster environment for SLURM managed compute cluster, sometimes you may want to override the
default auto-environment detection to default to lightning launcher.


--- Torch Elastic ---
Environment for fault-tolerate elastic training with `torchelastic`.

'''