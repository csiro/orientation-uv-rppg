import os
import torch
from lightning import LightningModule

from abc import ABC, abstractmethod

from typing import *
from torch import Tensor
from torch.nn import Module, ModuleList
from src.datamodules.datamodels import DataModel
from src.datamodules import DataModule
from src.metrics import MetricModule


class ModelModule(LightningModule, ABC):
    """

    Within this `ModelModule` we define and de-couple the stage-specific
    behaviour from the neural architecture and processing operations, to allow modularization
    of the architecture and information flow from the training, validation, testing, and
    prediction behaviour.

    Provided `network` defines the neural architecture and should typically only implement
    the forward pass logic.

    `training_step` defines the training-behaviour of the model, this include: loss
    calculation from the model to perform backpropagation on.

    `validation_step` defines the validation-behaviour of the model, this includes: loss
    calculation without gradient hooks disabled to allow for comparison with the training
    loss (`EarlyStopping` due to overfitting, best performing model, etc.), and optionally
    metric calculatiom calculation to allow for monitoring of the inference-mode behaviour
    of the model over the training process.

    `test_step` defines the test-behaviour of the model, this includes loss calculation,
    metric calculation, and summary calculation (e.g. `ConfusionMatrix`) to allow for more
    contextual understanding of the behaviour of the model. `DataLoader` can include 
    multiple sources such as the `train`, `validation`, and `test` populations to ensure
    comprehensive description of the performance.

    `predict_step` defines the deployment prediction-behaviour of the model, typically
    only includes computing and returning the predictions.

    
    --- LOGGING ---
    `Metric` logging in Lightning occurs through `self.log` and `self.log_dict` methods. Both
    support logging of scalar tensors.

    When `Metric` or `MetricCollection` objects are logged inside a `LightningModule` using the
    `self.log` method, the metrics will be logged based on the `on_step` and `on_epoch` flags
    present. If `on_epoch` is True, then logger automatically logs at the end of the epoch.

    When you call `self.log` with a key:val pair, it will make these metrics available to
    other callback functions such as `EarlyStopping` or any monitoring function.
    
    When using `on_epoch=True` should generally set `sync_dist=True` to accumulate metrics
    across devices, however may incur significant communication overhead.

    Decided to utilize `on_epoch=False` as `EarlyStopping` only makes a decision based on
    the validation loss, not the relative divergence.

    NOTE: `sync_dist` will have no impact on logged `Metric` objects, their contain their
    own internal distributed synchronization logic, stick to the default behaviour.

    NOTE: If you want to calculate epoch-level metrics use `log` for automatic accumulation
    across batches in an epoch, `log` also handles reduction of metrics across devices.

    NOTE: Recommended to directly log the `Metric` objects to allow flags in `self.log` to
    control when items are logged.
    

    --- OPTIMIZERS ---
    Optimizer, Multiple Optimizers, Manual/Automatic Optimization

    Optimizer should define the optimization scheme of 

    Default optimization behaviour:
        self.optimizers to access the optimizers returned from configure_optimizers
        lr_schedulers: s
            interval

            
    --- SCHEDULERS ---
    Learning rate schedulers

    
    --- HYPERPARAMETERS ---
    Standardized way of saving the information during training into checkpoints. 
    
    Allows you to automatically save all hyperparameters passed to `__init__` by calling 
    `save_hyperparameters`. 
    
    LightningModule when loaded also has access to the hyperparameters, you can override
    these when loading with `load_from_checkpoint(path, *args, **kwargs)`.

    You can restore the full training process using `trainer.fit(..., ckpt_path="...")`,
    which will automatically restore models, epochs, steps, optimizers, schedulers, etc.

    
    --- HOOKS ---
    x


    --- CALLBACKS ---
    Model-specific callbacks can be added through `configure_callbacks` and will extend the
    list provided to the Trainer, and REPLACE trainer callbacks if there are more than one
    of the same type.

    NOTE: `ModelCheckpoint` callbacks will ALWAYS run last to ensure all states are saved.

    """
    def __init__(
        self, 
        network: Module, 
        losses: Optional[MetricModule] = None,
        optimizers: Optional[Any] = None, 
        metrics: Optional[MetricModule] = None,
        summaries: Optional[MetricModule] = None,
        schedulers: Optional[Any] = None,
        name: Optional[str] = None,
        datamodule: Optional[DataModule] = None,
        process: Optional[Dict[str, Callable]] = None,
        writer: Optional[Callable] = None,
        *args, **kwargs
    ) -> None:
        """_summary_

        # TODO: Determine a better method of dynamically instantiating withotu reqs.

        Args:
            network (Module): _description_
            losses (MetricModule): _description_
            optimizers (Any): _description_
            metrics (MetricModule): _description_
            summaries (MetricModule): _description_
            schedulers (Optional[Any], optional): _description_. Defaults to None.
        """
        super(ModelModule, self).__init__()

        # Extract information from `DataModule`
        self.name = name
        self.network = network
        self.optimizer_factory = optimizers
        self.scheduler_factory = schedulers

        # Processing
        self.process_outputs = process.pop("outputs", None) if process is not None else None
        self.process_predictions = process.pop("predictions", None) if process is not None else None

        # MetricCollections
        '''
        NOTE: Metrics contain internal states to keep track of data they have been provided,
        these states are NOT added to the models `state_dict` by default. Given we restore 
        `ModelCheckpoint`s from the start of an epoch this behaviour acceptable.

        NOTE: Declared inside `LightningModule`
        '''
        mkwargs = datamodule.information() if datamodule is not None else {}
        self.losses = losses(**mkwargs) if losses is not None else None
        self.metrics = metrics(**mkwargs) if metrics is not None else None
        self.summaries = summaries(**mkwargs) if summaries is not None else None

        self.use_groups = "groups" in mkwargs.keys()

        # Writer
        '''
        Method for exporting the resulting prediction and detection information to disk
        '''
        self.writer = writer
    
    def forward(
        self, 
        inputs: Dict[str, DataModel]
    ) -> Dict[str, DataModel]:
        """ Perform a forward pass of the network, specific implementations may define
        different post-processing behaviours of the network output or loss calculation
        in training, validation, testing, or prediction specific behaviours. Hence, the
        forward pass should only define the architecture.

        Returned `Dict` should contain all necessary items to perform the `self.losses`
        computation in training mode and the `self.prediction` computation in prediction
        mode.
        """
        return self.network(inputs)


    # ===== [ TRAINING BEHAVIOUR OPERATIONS ] =====
    
    def compute_outputs(
        self,
        results: Dict[str, DataModel]
    ) -> Dict[str, DataModel]:
        """ Compute the training mode outputs of the model based on the forward pass
        results. Allows you to define training specific post-processing operations if
        desired.

        NOTE: We typically require that all processes here are `differentiable` to ensure
        that we can back-propagate to compute the gradients for the updates.

        NOTE: We do NOT require processes to be differentiable IF the model is NOT already
        differentiable. E.g. unsupervised signal processing methods such as `CHROM`.
        """
        return self.process_outputs(results)["data"] if self.process_outputs is not None else results

    def compute_loss(
        self,
        losses: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """ Compute the aggregated loss to be used for backpropagation, allows for 
        user-define aggregation schemes e.g. loss weighting.
        """
        return sum(loss for loss in losses.values())
    

    # ===== [ EVALUATION BEHAVIOUR OPERATIONS ] =====
    
    def compute_predictions(
        self,
        results: Dict[str, DataModel],
        *args, **kwargs
    ) -> Dict[str, DataModel]:
        """ Compute the prediction mode predictions of the model based on the forward
        pass results. Allows you to define prediction specific post-processing operations
        if dsired. Can simply contain `self.outputs(...)`.

        NOTE: Default `predict` mode behaviour is same as `training` mode behaviour.

        NOTE: Combined with post-processing this can be used to calculate the Heart-rate
        as often the evaluation mode behaviour will differ from the training mode behaviour.

        NOTE: 
        """
        return self.process_predictions(results)["data"] if self.process_predictions is not None else self.compute_outputs(results, *args, **kwargs)

    
    # ===== [ TRAIN BEHAVIOUR ] =====

    def training_step(
        self, 
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], 
        batch_idx: int
    ) -> Dict[str, Any]:
        """ Perform the training step of the `LightningModel` including the forward pass, loss
        computation, metric computation, logging, and returning the `loss` for backpropagation.

        NOTE: If you want to perform epoch-level operations such as utilizing all of the outputs
        from `training_step` then override the `on_train_epoch_end` method.
        """
        # Unpack batch and perform forward pass
        inputs, targets, source, batch_size = batch.unpack
        results = self(inputs)

        # Loss computation
        # TODO: Enable different training/validation dataloader configurations (dont want certain processing at train-time)
        outputs = self.compute_outputs(results)
        losses = self.losses("train", outputs, targets, None, False) # batch
        loss = self.compute_loss(losses) # batch_loss # TODO: Define `Sum` metric to combine losses
        
        # Logging : batch-level
        self.log_dict(losses, on_step=True, on_epoch=False, logger=True, sync_dist=False)
        self.log_dict({"train/losses": loss}, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=False)

        return {"loss": loss}

    def on_train_epoch_end(self, *args, **kwargs):
        """_summary_
        """
        # Compute epoch-level metrics : synchronized
        losses = self.losses.compute("train")
        loss = self.losses.group_fn(losses, self.compute_loss) # Apply compute_fn to groups

        # Reset metrics
        self.losses.reset("train")

        # Log : even though we don't really care about epoch-level training loss
        self.log_dict({**loss, **losses}, on_step=False, on_epoch=True, logger=True, sync_dist=False)
    

    # ===== [ VALIDATION BEHAVIOUR ] =====

    def validation_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int
    ) -> Dict[str, Any]:
        """
        During the validation step we have access to ground-truths and we are in evaluation
        mode.

        NOTE: It's recommended to perform validation on a single device to ensure each sample only
        gets evaluated once to ensure repeatable benchmarking, in a multi-device setting samples can
        be duplicated to ensure even batch sizes across devices.

        NOTE: When running in distributed mode, we have to ensure that the validation and test step
        logging calls are synchronized across processes, this is done with `sync_dist=True` in the
        validation and testing stop logging calls. However, if you use any built-in metrics or custom
        metrics with `torchmetrics` these do NOT need to be updated and are automatically handled
        for you.
        """
        # Unpack batch and perform forward pass
        inputs, targets, groups, batch_size = batch.unpack
        results = self(inputs)

        # Loss computation
        outputs = self.compute_outputs(results)
        self.losses("valid", outputs, targets, groups if self.use_groups else None, True)
        # losses = self.losses("valid", None, True, outputs, targets) # epoch-level
        # loss : Tensor = self.compute_loss(losses) # compute batch-level loss to provide to `self.log`

        # Metric computation
        predictions = self.compute_predictions(results)
        self.metrics("valid", predictions, targets, groups if self.use_groups else None, True) # epoch-level

        # Log : accumulate epoch-level `valid_loss` to be available to callbacks (N/A in `on_valid_epoch_end`)
        # self.log_dict({"valid_loss": loss}, on_step=False, on_epoch=True, prog_bar=False, logger=False)
    
    def on_validation_epoch_end(self, *args, **kwargs):
        """_summary_
        """
        # Compute epoch-level metrics
        losses = self.losses.compute("valid")
        metrics = self.metrics.compute("valid") 
        loss = self.losses.group_fn(losses, self.compute_loss) # Apply compute_fn to groups

        # Reset metric states
        self.losses.reset("valid")
        self.metrics.reset("valid")

        # Log epoch-level : should be same as valid_loss
        self.log("valid/loss", loss["valid/losses"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        self.log_dict({**loss, **losses, **metrics}, on_step=False, on_epoch=True, logger=True, sync_dist=False)


    # ===== [ TEST BEHAVIOUR ] =====

    def test_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int
    ) -> Dict[str, Any]:
        '''
        During the testing step we have access to ground-truths and we are in evaluation mode.

        We want to compute the testing loss and the testing predictions, then compare these
        to the ground-truths to obtain the testing metrics.

        We additionally want to log specific media for visual inspection.

        NOTE: Duplicate `validation_step` behaviour essentially but with a different flagged stage.
        
        '''
        # Unpack and perform forward pass
        inputs, targets, groups, batch_size = batch.unpack
        results = self(inputs)

        # Loss computation
        outputs : Dict[str, DataModel] = self.compute_outputs(results)
        losses = self.losses("test", outputs, targets, groups if self.use_groups else None, False)

        # Metric & summaries computation
        predictions : Dict[str, DataModel] = self.compute_predictions(results.copy()) # TODO: Remove copy op and shift to Pipe process
        metrics = self.metrics("test", predictions, targets, groups if self.use_groups else None, False)
        # self.summaries("test", None, predictions, targets) # TBD

        # Logging to file-system
        if self.writer is not None: self.writer(inputs, targets, groups, outputs, predictions, losses, metrics)
    
    def on_test_epoch_end(self, *args, **kwargs):
        """_summary_
        """
        # Compute epoch-level metrics
        losses = self.losses.compute("test")
        metrics = self.metrics.compute("test")
        loss : Dict[str, Tensor] = self.losses.group_fn(losses, self.compute_loss) # Apply compute_fn to groups

        # Reset metric states
        self.losses.reset("test")
        self.metrics.reset("test")

        # Log
        self.log_dict({**loss, **losses, **metrics}, on_step=False, on_epoch=True, logger=True, sync_dist=False)
    

    # ===== [ PREDICTION BEHAVIOUR ] =====
    
    def predict_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int
    ) -> Dict[str, Any]:
        """
        During the prediction step we do NOT have access to ground-truths and are in evaluation
        mode.

        We want to compute the predictions and return these.
        """
        # Unpack and perform forward pass
        inputs, _, _, _ = batch.unpack
        results = self(inputs)

        # Compute predictions
        predictions = self.compute_predictions(results)

        return predictions


    # ===== [ MODULE CONFIGURATION ] =====

    def configure_optimizers(
        self, 
        *args, **kwargs
    ) -> Union[str, Any]:
        '''
        Define the optimizers and learning-rate schedulers to utilize in your optimization

        NOTE: You can override the default automatic optimization behaviour to implement your
        own more complex optimization schemes if desired, this is useful for systems such as GANs.
        see: lightning manual_optimization

        Returns:
            Single Optimizer
            List/Tuple of Optimizers
            Tuple of Lists (Optimizers, Schedulers)
            Dictionary with optimizer and lr_scheduler keys

        NOTE: 
            Lightning calls `backward` and `step` during automatic optimization
            If specify scheduler  key
            Lightning will automatically handle the optimizer with precision
            Lightning handles closure function for optimizers such as `LBFGS`
        
        NOTE: If you utilize multiple optimizers, then you will have to enable `manual_optimization`
        network and step them yourself as required.

        NOTE: If you have to control how often the optimizer steps, then you should override the
        `optimizer_step` hook.

        TODO: Implement support for multiple optimizers and learning-rate schedulers (framework done).

        '''
        # Construct optimizers
        optimizers = self.optimizer_factory(self.network)
        config = {"optimizer": list(optimizers.values())[0]} # NOTE: Currently only supports a single optimizer

        # Construct schedulers
        schedulers = self.scheduler_factory(optimizers) if self.scheduler_factory is not None else None
        if schedulers is not None: config = {**config, **{"lr_scheduler": list(schedulers.values())[0]}} # NOTE: Currently only supports a single scheduler

        return config

    # def configure_callbacks(
    #     self,
    #     *args, **kwargs
    # ) -> Union[Sequence[Any], Any]:
    #     '''
    #     https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-callbacks
    #     '''
    #     pass

    def information(self, batch_size: Optional[int] = 256) -> Dict[str, Any]:
        """ Compute and return information used for logging and instantiation.

        Child-class can override and append additional information to the returned
        Dict if desired.
        
        Returns:
            Dict[str, Any]: _description_
        """
        # TODO: Compute FLOPS.
        # from math import floor, log2

        num_params = sum([p.numel() for p in self.network.parameters()])
        num_grad_params = sum([p.numel() for p in self.network.parameters() if p.requires_grad])
        # batch_sizes = [2**idx for idx in range(floor(log2(batch_size))+1)]  
        # dummy_size = list(self.network.dummy_input(1)[0].squeeze(0).size())      
        
        # inference_time_cpu = batch_size_timeit(self.network, batch_sizes, "cpu")
        # inference_time_gpu = batch_size_timeit(self.network, batch_sizes, "gpu")



        return {
            "num_parameters": num_params,
            "num_parameters_gradient": num_grad_params,
            # "batch_size": batch_sizes,
            # "dummy_size": dummy_size,
            # "inference_time_cpu": inference_time_cpu,
            # "inference_time_gpu": inference_time_gpu
        }


# TODO: Method for computing timing based metrics of model on cpu, gpu, compile, torchscript, etc.
def batch_size_timeit(network: Module, batch_sizes: List[int], device: Optional[str] = "cpu") -> float:
    import timeit
    if device == "cpu":
        network.cpu()
    elif device == "gpu":
        network.cuda()
    else:
        raise ValueError(f"Unhanled device string specified ({device}). Please use 'cpu' or 'gpu'")
    results = []
    for batch_size in batch_sizes:
        if device == "cpu":
            input = network.dummy_input(batch_size).cpu()
            result = timeit.repeat(lambda: network(input), number=10)
            results.append(sum(result)/len(result))
        else:
            input = network.dummy_input(batch_size).cuda()
            t = 0
            n = 10
            for _ in range(n):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                network(input)
                end.record()
                torch.cuda.synchronize()
                t += start.elapsed_time(end)
            results.append(t/n)

    return results
