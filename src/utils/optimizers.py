from src.utils import Factory

from typing import *
from torch.nn import Module
from torch.optim import Optimizer


class OptimizerFactory(Factory):
    """
    With more complex training setups, you may have multiple losses for different components
    of the model, and you want different optimizers for these different components. Should 
    implement each of these in a modular manner with their own wrapped model, optimizer, and
    metrics.
    """
    def __init__(self, optimizers: Dict[str, Callable], *args, **kwargs) -> None:
        '''
        '''
        # Fully initialize the constructor
        super(OptimizerFactory, self).__init__(optimizers, *args, **kwargs)

    def __call__(self, model: Module, *args, **kwargs) -> Dict[str, Optimizer]:
        '''
        '''
        # Provide generator of parameters to consume per optimizer
        return {
            name: constructor(model.parameters(), *args, **kwargs) 
            for (name, constructor) in self.constructors.items()
        }

