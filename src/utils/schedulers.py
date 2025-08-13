from dataclasses import dataclass

from src.utils import Constructor, Factory

from typing import *
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class SchedulerFactory(Factory):
    """
    """
    def __init__(self, schedulers: Dict[str, Callable], *args, **kwargs) -> None:
        '''
        '''
        super(SchedulerFactory, self).__init__(schedulers,*args, **kwargs)

    def __call__(self, optimizers: Dict[str, Optimizer], *args, **kwargs) -> Dict[str, LRScheduler]:
        '''
        '''
        return {
            f"{name}_{constructor.optimizer}": constructor(
                optimizers[constructor.optimizer], 
                *args, **kwargs
            ) 
            for (name, constructor) in self.constructors.items()
        }


@dataclass
class Configuration:
    scheduler: LRScheduler
    interval: Optional[str] = "epoch"
    frequency: Optional[int] = 1
    monitor: Optional[str] = None
    strict: Optional[bool] = True
    name: Optional[str] = None


class SchedulerConfiguration:
    """
    """
    def __init__(self, scheduler: Callable, configuration: Callable, optimizer: str) -> None:
        ''' 
        '''
        self.scheduler = scheduler
        self.configuration = configuration
        self.optimizer = optimizer

    def __call__(self, optimizer: Optimizer, name: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        '''
        '''
        return vars(self.configuration(
            scheduler = self.scheduler(optimizer),
            name = name
        ))
    

class SchedulerConfigurationFactory(Factory):   
    """
    """ 
    def __init__(self, schedulers: Dict[str, Callable], *args, **kwargs) -> None:
        '''
        '''
        super(SchedulerConfigurationFactory, self).__init__(schedulers,*args, **kwargs)

    def __call__(self, optimizers: Dict[str, Optimizer], *args, **kwargs) -> Dict[str, LRScheduler]:
        '''
        name : string defining the SchedulerConfigurationConstructor
        constructor : SchedulerConfigurationConstructor
        constructor.optimizer : string defining the optimizer key
        '''
        return {
            f"{name}_{constructor.optimizer}": constructor(
                optimizer=optimizers[constructor.optimizer], 
                name=f"{name}_{constructor.optimizer}", 
                *args, **kwargs
            ) 
            for (name, constructor) in self.constructors.items()
        }
