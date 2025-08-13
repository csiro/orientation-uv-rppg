from torch.nn import Module
from abc import ABC, abstractmethod

class Network(Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Network, self).__init__()

    @abstractmethod
    def dummy_input(self, batch_size: int) -> None:
        pass
