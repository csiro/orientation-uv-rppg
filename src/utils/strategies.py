import os
import subprocess
from torch import distributed
from contextlib import ContextDecorator

from typing import *         


def popen_global_rank_zero(command: List[str]) -> None:
     node_rank = int(os.environ.get("NODE_RANK", 0))
     local_rank = int(os.environ.get("LOCAL_RANK", 0))
     if (node_rank == 0) and (local_rank == 0):
          p = subprocess.Popen(command)
          p.wait()
