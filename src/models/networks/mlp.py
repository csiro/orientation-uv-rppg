import torch
from src.models.networks import Network
from torch import nn

from typing import *
from torch import Tensor


class MLP(Network):
    def __init__(self) -> None:
        super(MLP, self).__init__()
        # Attributes
        num_classes = 10
        dims = (1, 28, 28)
        channels, width, height = dims
        hidden_size = 64

        # Model
        self.model = nn.Sequential(OrderedDict([
            ("flatten1", nn.Flatten()),
            ("linear1", nn.Linear(channels * width * height, hidden_size)),
            ("relu1", nn.ReLU()),
            ("linear2", nn.Linear(hidden_size, num_classes))
        ]))

    def forward(self, inputs: Dict[str, Tensor]) -> Any:
        return self.model(inputs)
    
    def dummy_input(self, batch_size: int) -> Tensor:
        return torch.rand(size=(batch_size,1,28,28))