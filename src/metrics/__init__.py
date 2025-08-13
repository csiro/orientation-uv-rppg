'''
=== [ TorchMetrics ] ===

Metrics API provides `update`, `compute`, and `reset` functions to the user, allows
for calling similar to `torch.nn.Moduel` using `metric(...)` performing a forward
pass. The `forward` method of `Metric` serves dual purpose of calling `update` on
its input and returning the metric OVER the provided input.

Work with `DDP` in `torch` and `lightning` by default, when `compute` is called in
distributed mode the internal state of each metric is synchronized AND reduced
acros each process, SUCH THAT, logic in `compute` is applied to identical state info
across all processes.

NOTE: Do NOT mix metric states across training, validation, and testing, recommended
to re-initialize the metric per mode.

Example::
    train_acc, valid_acc = BinaryAcc(...), BinaryAcc(...)
    for ... in train_dataloader:
        ...
        batch_acc = train_acc(preds, targets)

    for ... in valid_dataloader:
        ...
        valid_acc.update(preds, targets)

    total_train = train_acc.compute()
    print(f"Train: {total_train}")

    total_valid = valid_acc.compute()
    print(f"Valid: {total_valid}")

    train_acc.reset()
    valid_acc.reset()

NOTE: Metrics contain internal state to keep track of data they have been provided.

NOTE: Metric states are NOT added to the models `state_dict` by default, to change
this, AFTER initializing the metric use `.persistent(mode=True)` to enable this
behaviour (allows loadining of state from checkpoint during training).

NOTE: Due to specialized logic around metric states, do NOT recommended nested
metrics, instead subclass a metric or use `MetricCollection`.


--- Devices ---
Subclass `torch.nn.Module` and their metric states behave similar to buffers and
parameters, hence metric states should be moved to same device as the input of the
metric.

When PROPERLY DEFINED inside a `LightningModule` this will be done automatically,
being properly defined means being a child module of the model `.children()`.

Hence, cannot be placed in `list` or `dict`, instead use `ModuleList` or `ModuleDict`
instead, or alternatively use `MetricCollection`.

With `DistributedDataParallel` you should be aware that DDP will add additional
samples to the dataset to ensure a balanced `batch_size` across all `num_processors`.

The added samples WILL be replicates, hence will un-balance the calclated metric
value towards the duplicated sample, leading to a WRONG result. During training or
validation that may not be important, however testing behaviour should be setup to
only occur on a single GPU or employ a `join` context.


--- Precision ---
Most metrics can be used with `torch.half` precision, however:

`pytorch` has better supprot for 16-bit precision much earlier on GPU than CPU, hence
recommend if you want to use metrics with half precision on CPU to upgrade to version
`pytorch>=1.6.x`

Some metrics do NOT work in half precision on CPU, this will be explicitly stated in
the docstring, but includes: `PSNR`, `SSIM`, and `KL Divergence`.


--- Arithmetic ---
Supports most built-in operators for arithmatic, logic, and bit-wise operations, see
below for full list.

https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-arithmetics


--- Collection ---
...


--- Module vs. Functional ---
Functional metrics : function-based

Module metrics : class-based
Characterized by additional internal state which offers additional functionality
beyond the raw computation of functional metrics, including:
    Accumulation of multiple batches
    Automatic synchronization across devices
    Metric arithmetic

Example::
    for ... in enumerate(dataloader):
        preds = model(...)
        acc = metric(preds, targets)

    acc = metric.compute()
    print(acc)
    metric.reset()


--- Differentiability ---
Support backpropagation if all computations in the metric calculation are
differentiable, all module classes have `is_differentiable` to allow code to determine
if differentiable or not.

NOTE: Cached state is detached from the computational graph, and hence cannot be
backpropagated

`val = metric(preds, targets)` can be backpropagated, but `val = metric.compute()`
cannot.


--- Advanced Settings ---
You may run out of GPU VRAM if computing metrics on the GPU, you can use the flag
`compute_on_cpu` to automatically move metric states to the CPU after calling `update`
to reduce GPU memory usage. However, `compute` method will then be called on the CPU
instead of the GPU.

If running in a distributed environment, synchronization of metrics will automatically
be taken care of, however you can override these processes if desired.

`dist_sync_on_step` to indicate whether metric should be synchronized each `forward`
call, not recommended due to synchronization cost.

`process_group` default is to synchronize across the world (all processes), however
you can provide `torch._C._distributed_c10d.ProcessGroup` to override this with
specific devices.

`dist_sync_fn` default is to perform `torch.distributed.all_gather()` to perform the
synchronization between devices, however you can provide another callable function
to perform custom synchronization.

    
--- Custom Metrics ---
Subclass `Metric`
Implement `__init__` and add `self.add_state` for internal states
Implement `update` for updating metrics
Implement `compute` for final computation

See: https://torchmetrics.readthedocs.io/en/stable/pages/implement.html#implement


--- Lightning ---
Offer additional benefits

Module metrics will automatically be placed on the correct device when PROPERLY
DEFINED inside a `LightningModule`, will perform the `.to(device)` call for you.

- Logging -
Logging metrics in lightning using the `self.log` inside a `LightningModule`. Logging
can be done in two ways, logging of the `Metric` object directly or via. the computed
value.

When `Metric` objects which return a scalar tensor are logged directly inside your
`LightningModule` using `self.log` then will log the metric based on `on_step` and
`on_epoch` flags present. If `on_epoch=True` then logger automatically calls `compute`
at the end of the epoch before logging.

You can also manually log the output of the metrics e.g. using the `training_epoch_end`
hook inside the `LightningModule` with `self.log("acc_epoch", self.acc.compute())`.
Doing it this way requires you to manually reset the metrics at the end of the epoch
using `.reset()`. Generally, recommended to stick to logging the `Metric` directly,
and avoid mixing the two methods.

NOTE: `sync_dist`, `sync_dist_op`, `reduce_fx`, etc. flags in `self.log` do NOT affect
the metric logging in any manner, the metric class contains its own distributed 
synchronization logic. However, this only applies for `Metric` and not functional
metrics.

NOTE: Whilst most metrics are scalar tensors, some such as `ConfusionMatrix` are
non-scalar and should be dealt with seperately.

Metric `reset` methods will automatically be called at the end of an epoch.


--- Common Pitfalls ---
Modular metrics contain internal states that should belong to only ONE `DataLoader`, in
case you are using multiple `DataLoader`s its recommended to seperately initialize a
modular metric instance for each `DataLoader` and use them seperately. Same holds for
using seperate metrics for training, validation, and testing.

Exmaple::
    self.val_acc = nn.ModuleList([Accuracy(...) for _ in range(2)])

    def val_dataloader(self) -> :
        return [DataLoader(...), DataLoader(...)]

    def validation_step(self, batch, batch_idx, dataloader_idx) -> :
        self.val_acc[dataloader_idx](preds, y)
        self.log("val_acc", self.val_acc[dataloader_idx])

Calling `self.log("val", self.metric(preds, targets))` as `self.metric` refers to the
forward method, and will return a tensor NOT the `Metric` object.

'''
import torch
import logging
from pathlib import Path
from functools import partial
from torchmetrics import Metric, MetricCollection
from torch.nn import Module, ModuleList, ModuleDict
from src.datamodules.datamodels import DataModel
from importlib import import_module

from typing import *
from torch import Tensor

log = logging.getLogger(__name__) # Root logger


def DynamicMetric(metric: str, *args, **kwargs) -> Metric:
    """ Dynamically construct and instantiate a `Metric` with additional parsing of the 
    desired `prediction` and `target` into the `Tensor`s for processing.

    Args:
        metric (Metric): `Metric` to inherit from.

    Returns:
        Metric: _description_
    """
    parts = metric.split(".")
    module, package = ".".join(parts[:-1]), parts[-1]
    metric = getattr(import_module(module), package)

    class DynamicMetric(metric):
        def __init__(self, output_loc: str, target_loc: str, use_data: Optional[bool] = True, *args, **kwargs) -> None:
            super(DynamicMetric, self).__init__(*args, **kwargs)

            # # Outputs/predictions key
            # if "." in output_loc:
            #     output_loc, self.output_attr = tuple(output_loc.split("."))
            # else:
            #     self.output_attr = None

            # if "/" in output_loc:
            #     self.output_group, self.output_key = tuple(output_loc.split("/"))
            # else:
            #     self.output_group = None
            #     self.output_key = output_loc

            # # Targets key
            # if "." in target_loc:
            #     target_loc, self.target_attr = tuple(target_loc.split("."))
            # else:
            #     self.target_attr = None

            # if "/" in target_loc:
            #     self.target_group, self.target_key = tuple(target_loc.split("/"))
            # else:
            #     self.target_group = None
            #     self.target_key = target_loc


            # Output location
            self.output_loc = output_loc
            if "/" in self.output_loc:
                self.output_key = str(Path(self.output_loc).parent)
                self.output_attr = str(Path(self.output_loc).name)
            else:
                self.output_key = self.output_loc
                self.output_attr = None

            # Target location
            self.target_loc = target_loc
            if "/" in self.target_loc:
                self.target_key = str(Path(self.target_loc).parent)
                self.target_attr = str(Path(self.target_loc).name)      
            else:
                self.target_key = self.target_loc
                self.target_attr = None

            # Use data of DatasetSample ( should be inferred from none)
            self.use_data = use_data

        def update(self, outputs: DataModel, targets: DataModel, *args, **kwargs) -> None:
            # # Extract dynamic data from `DataModel` for metric calculation
            # if self.output_attr is not None:
            #     if self.output_group is not None:
            #         outputs = outputs[self.output_key].data
            #         targets = targets[self.target_ket].data
            #     else:
            #         outputs = outputs[self.]

            
            
            if self.use_data:
                outputs = outputs[self.output_key].data
                targets = targets[self.target_key].data
            else:
                outputs = outputs[self.output_key].attrs[self.output_attr]
                targets = targets[self.target_key].attrs[self.target_attr]

            # Update via. super method
            super(DynamicMetric, self).update(outputs, targets, *args, **kwargs)

    return DynamicMetric(*args, **kwargs)


class MetricModule(Module):
    """ Acts as a container for stage-specific and context-specific `MetricCollection`.

    Framework will return an associated `group` for each `batch` item returned by the
    `DataLoader`, this may NOT be able to be determined ahead of time, such as with 
    iterable-style `Datasets`.

    Hence the `MetricCollectionContainer` should contain the necessary logic to
    dynamically instantiate and append dynamically contained `MetricCollection`s to
    allow us to calculate `Metric`s on a aggregate or per-group basis for contextual
    analysis.

    NOTE: We expect this to incur significant overhead due to the potentially large 
    number of `update` and `compute` calls incurred.

    We define the `MetricCollectionContainer` as a `Module` to:
        Ensure it is included in the `state_dict` of the attached `LightningModule` to 
        allow resuming from `ModelCheckpoint`s

    When defined inside a `LightningModule` there are additional benefits:
        Ensure `Metrics` are placed on the same device as data when defined inside a
        `LightningModule`

        Native support for logging metrics using `self.log` inside `LightningModule`

        `reset` method will automatically be called at the end of an epoch.

    NOTE: `MetricCollection` will try to reduce computations needed for metrics in the 
    collection by checking their `compute_group`. Can significantly speed up calculation, 
    this feature is only available when calling `update` and NOT `forward` due to the 
    internal `forward` logic.

    compute_groups=False + forward (48.9)
    Function Took 0.0743 seconds
    Function Took 1.2147 seconds
    Function Took 1.6581 seconds
    Function Took 1.8279 seconds

    compute_groups=True + forward (60.9)
    Function Took 0.1523 seconds _duplicates
    Function Took 1.7490 seconds _nonscalar
    Function Took 2.0181 seconds _nested_group
    Function Took 1.9876 seconds _flat_group

    compute_groups=False + update/compute (61.6)
    Function Took 0.0686 seconds
    Function Took 1.1109 seconds
    Function Took 1.6548 seconds
    Function Took 1.5521 seconds

    compute_groups=True + update/compute (73.0)
    Function Took 0.0113 seconds
    Function Took 1.7265 seconds
    Function Took 1.7813 seconds
    Function Took 1.5526 seconds

    Args:
        Module (_type_): _description_
    """
    delimeter = "/"

    def __init__(self, context: str, metrics: Dict[str, Metric], *args, **kwargs) -> None:
        super(MetricModule, self).__init__()
        self.context = context

        # Group-agnostic/specific kwargs
        # TODO: Some metric collections require additional input information (how to parse out un-needed args dynamically, maybe just return a subject that wil lactually be used but then still need to pass dynamically...)
        self.agnostic = MetricCollection({name: metric() for name, metric in metrics.items()})
        self.specific = MetricCollection({name: metric() for name, metric in metrics.items()})

        self.container = ModuleDict()

        # TODO: Some metrics inherently require the number of classes : Computing average accuracy per class may not always make sense
        # log.warning(f"Caution when computing group-level metrics. E.g. Computing AverageAccuracy with nc=10 when the ng=1 will result in AverageAccuracy being 1/nc as averaging across classes.")

        '''
        some Metric
            when we instantiate : need to pass parameters...
                instnaitation may be context dependent... e.g. number of classes or groups
        
        what we want

        Pass dict of metrics to utilize
        {
            "name": Metric
        }

        During setup we compute contextual parameters e.g.
        # TODO: Multiple mutually inclusive groups : avoid re-running
        datainfo = {
            "groups": ...
            "classes": ...
        }

        During instantiation we'll want to reference these...
        Metric(*args, **kwargs, num_classes=len(groups))

        Global-level statistics: Depends on the population of the resultant dataset labels used during training : 
        Local-level statistics: Depends on the population of the groups of dataset labels 
        '''

    def forward(self, 
        stage: str,
        outputs: Tensor,
        targets: Tensor,
        groups: Optional[List[str]] = None, 
        only_update: Optional[bool] = False, 
        *args, **kwargs
    ) -> Any:
        """_summary_

        NOTE: When `compute` is called in `DDP` mode the internal state of each metric is
        synchronized and reduced across each process to `compute` is applied to identical
        state information from all processes.

        NOTE: It's important to track the aggregate `Metric` alongside, computing the 
        epoch-level `Metric`s, as averaging is dependent on the number of samples, whilst
        this will double the memory usage it allows independent accumulation.

        Args:
            stage (str): _description_
            group (Optional[str], optional): _description_. Defaults to None.

        Returns:
            Any: _description_
        """
        # Dynamically intantiate group-agnostic `MetricCollection`
        key = f"{stage}{self.delimeter}{self.context}{self.delimeter}" 
        if key not in self.container:
            self.container.add_module(key, self.agnostic.clone(prefix=key))
        
        # Conditionally `update` internal-state or perform `forward` on batch
        if only_update:
            self.container[key].update(outputs, targets, *args, **kwargs) # update internal state
            result = None
        else:
            result = self.container[key](outputs, targets, *args, **kwargs) # compute for provided batch ONLY

        # Dynamically instantiate group-specific `MetricCollection`
        '''
        Want to `update` a given `MetricCollection` with the `group`-specific `batch` data when desired,
        rather than just one-by-one.
        '''
        if groups is not None:
            for group in set(groups): # Process per group
                # Dynamically instantiate group-spepcifc `MetricCollection`
                key = f"{stage}{self.delimeter}{self.context}{self.delimeter}{group}{self.delimeter}" 
                if key not in self.container:
                    self.container.add_module(key, self.specific.clone(prefix=key))

                # Extract subset of outputs/targets for the current group
                # Computing masks seems to be the fastest way to do this 95it/s vs 70 it/s @ default with b128
                mask = [_grp == group for _grp in groups]
                if type(targets) == dict:
                    _outputs = {key:val[mask] for (key,val) in outputs.items()} # index Dict items
                    _targets = {key:val[mask] for (key,val) in targets.items()} 
                elif type(targets) == list:
                    _outputs = [val for (use, val) in zip(mask, outputs) if use] # index list of items
                    _targets = [val for (use, val) in zip(mask, targets) if use] 
                elif type(targets) == Tensor: 
                    _outputs = outputs[mask] # index Tensor
                    _targets = targets[mask] 
                else:
                    raise TypeError(f"Unhandled type for masking group: {type(targets)}")

                # Only ``update`` the group-specific containers : never backpropogate on them.
                self.container[key].update(_outputs, _targets, *args, **kwargs) # update internal state

        return result

    def compute(self, key: Optional[str] = None) -> Dict[str, Tensor]:
        """_summary_

        Returns:
            Dict[str, Tensor]: _description_
        """
        flat_results = {}
        for collection in self.container.values(): # for each `MetricCollection`
            if key is not None: # conditionally skip
                if key not in collection.prefix:
                    continue
            results = collection.compute() # compute (includes DDP synchronization)
            for name, val in results.items(): # for each `Metric`
                flat_results[name] = val
        return flat_results
    
    def group_fn(self, items: Dict[str, Tensor], apply_fn: Callable) -> Dict[str, Tensor]:
        """_summary_

        Args:
            apply_fn (Callable): _description_

        Returns:
            Dict[str, Tensor]: _description_
        """
        # Group items
        '''
        Will produce a large number of groups with groups included; but allows for contextual insights.
        '''
        groups = {self.delimeter.join(k.split(self.delimeter)[:-1]) for k in items.keys()}
        grouped_items = {g: {} for g in groups}
        for g, i in grouped_items.items():
            for k, v in items.items():
                if g == self.delimeter.join(k.split(self.delimeter)[:-1]):
                    _k = k.split(self.delimeter)[-1]
                    i[_k] = v

        result_items = {}
        for group, items in grouped_items.items():
            result_items[group] = apply_fn(items) #sum([val for val in _items.values()])

        return result_items
    
    def reset(self, key: Optional[str] = None) -> None:
        """_summary_
        """
        for collection in self.container.values(): # iterate over `MetricCollection`
            if key is not None: 
                if key not in collection.prefix: # conditionally skip if NOT in
                    continue 
            collection.reset()
