import torch

from typing import *


class CUDA:
    def __init__(self) -> None:
        self.available = torch.cuda.is_available()
        if not self.available: raise ValueError(f"CUDA is not available.")

        self.devices = None

    def balance_devices(self, index: int, jobs: int, devices_per: int, vram_per: float) -> int:
        """
        """
        # Validate input arguments
        device_count = torch.cuda.device_count()
        assert devices_per <= device_count, f"Require devices ({devices_per}) <= device_count ({device_count})"
        
        # Create list of device indexes if first time entering
        if self.devices == None:
            self.devices = (list(range(device_count)) * jobs)[::-1]

        # Pop items from list to use
        use_devices = [self.devices.pop() for _ in range(devices_per)]

        # Handle memory requirements (limit each process to fraction)
        # for device in use_devices:
        #     torch.cuda.fract
        
        return use_devices
