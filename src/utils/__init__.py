from functools import partial
from importlib import import_module
from abc import ABC, abstractmethod

from typing import *


def load_fn(
    call: str,
    *args, **kwargs
) -> Any:
    '''
    Example::
        >>> model = NeuralNetwork(...)
        >>> opt_constructor = dynamic_call("torch.optim", "Adam", True, lr=0.02)
        >>> optimizer = foo(model.parameters())
    '''
    _parts = call.split(".")
    module, func = ".".join(_parts[:-1]), _parts[-1]
    return getattr(import_module(module), func)


class Constructor(ABC):
    """

    NOTE: We could implement the constructor per child class to define specific 
    partial instiantiation logic, however, we've decided to leave this to the
    hydra configuration files.
    """
    def __init__(
        self, 
        call: Union[str, Callable],
        partial_init: Optional[bool] = True,
        *args, **kwargs
    ) -> None:
        '''
        Partially initialise the provided call which can either be a direct function
        call or a string specifying a source module.
        '''
        # Dynamically load the function call
        if isinstance(call, str):
            call = load_fn(call)

        # Fully/partially initialise the call
        self.partial_init = partial_init
        self.call = partial(call, *args, **kwargs) if partial_init else call(*args, **kwargs)
    
    def __call__(self, *args, **kwargs) -> Any:
        '''
        Typically when overriding this class you will define some construction specific
        logic if required, however default behaviour is generally fine.
        '''
        return self.call(*args, **kwargs) if self.partial_init else self.call


class Factory:
    """
    """
    def __init__(self, constructors: Dict[str, Constructor], *args, **kwargs) -> None:
        '''
        '''
        self.constructors = constructors
    
    def __call__(self, *args,**kwargs) -> Dict[str, Any]:
        '''
        Typically when overriding this class you will define some factor specific
        cosntruction logic if required, however default behaviour is generally fine.

        Cases include when the constructor consumes a generator per item such as with
        torch.optim.Optimizer instantiation.
        '''
        return {
            key: constructor(*args, **kwargs)
            for (key, constructor) in self.constructors.items()
        }


from omegaconf import DictConfig
from hydra.utils import instantiate as hydra_instantiate

def instantiate_items(cfg: DictConfig, order_keys: Optional[List[str]] = None, apply_fn: Optional[Callable] = None) -> Dict[str, Any]:
    _items = {}

    # Instantiate specified keys first
    if order_keys is not None:
        for key in order_keys:
            if key not in cfg.keys():
                raise ValueError(f"Specified key={key} is not in config_keys={cfg.keys()}")
            _items[key] = hydra_instantiate(cfg[key])
            apply_fn(key, _items[key]) # 

    # Remaining keys are unordered : 
    remaining_keys = [key for key in cfg.keys() if key not in order_keys]
    for key in remaining_keys:
        _items[key] = hydra_instantiate(cfg[key])

    # Return instantiated items
    return _items

