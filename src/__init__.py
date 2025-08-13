import random
from math import floor
from omegaconf import OmegaConf
from multiprocessing import cpu_count

# Dynamic Resolution
OmegaConf.register_new_resolver("loss_name", lambda v:f"{v}/loss") # Dynamically define loss name : allow quick changing of configuration files
OmegaConf.register_new_resolver("dynamic_steps_per_epoch", lambda num_samples, batch_size: floor(int(num_samples) / int(batch_size))) # Dynamically calculate number of optimizer steps per epoch
OmegaConf.register_new_resolver("dynamic_pin_memory", lambda acc: acc not in ["cpu"]) # pin to paged memory if NOT using the CPU
OmegaConf.register_new_resolver("dynamic_batch_size", lambda b_sz, n_dev: int(b_sz) // int(n_dev)) # effective batch size is always b_s (need to be able to split evenly)

# General Arithmetic
OmegaConf.register_new_resolver("random_float", lambda x: random.random())
OmegaConf.register_new_resolver("random_int", lambda x, y: random.randint(int(x), int(y)))
OmegaConf.register_new_resolver("multiply", lambda x, y: float(x) * float(y))
OmegaConf.register_new_resolver("add", lambda x, y: float(x) + float(y))
OmegaConf.register_new_resolver("divide", lambda x, y: float(x) / float(y))
