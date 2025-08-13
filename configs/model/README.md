## Model
Each configuration contained in `/model` describes a configuration setup for a given model.

This includes both:
- Model-specific architecture (e.g. CHROM)
- Model-specific hyper-parameters (e.g. image_size, frequency cut-offs, etc.)
- Model-specific input processing (transform a standard sample into the required form) (e.g. normalize frame difference, RGB trace computation)
- Model-specific target processing (processing applied to target to enable loss and metric calculation) (e.g. signal processing, peak frequency detection)

Note, whilst this may result in more complex configuration files relating to the model, it enables us to de-couple the processing into dataset-specific and model-specific behaviours, improving the maintainabiltiy for a larger number of different configurations as the number of datasets and models expands.

We define dataset-specific processing configuration in a given dataset, this often includes which keys to reference for ACCESS `frames` or `labels` to de-couple this logic from the model. E.g. `inputs/traces` 

We define model-specific processing configuration in a given model, this often include which keys to reference for PLACING data for the model to use. E.g. `traces` tells the processing where to place the results for the model and the model where to access the data in the inputs.

Note, this is contingent in using a pre-established structure for the inputs, targets, and outputs, which we have outlined below.

Samples contained in a `dataset` will have the format `{sample_uuid: str, data : Dict[str, Any]}` where `data` has the format `{inputs/... : ..., targets/... : ...,}`. This is the format which `DataPipe` processes will operate on, THUS keys for processing should be represented in this format e.g. `inputs/<key>` or `targets/<key>`. 

Samples returned from the `dataloader` will contain batched versions of the samples, and thus will have the format `{inputs/... : ..., targets/... : ...}` with an additional dimension along `dim=0` representing the batch samples.

Samples are unpacked into a tuple when being used within the training loop, and hence will be provided to the model in the format `inputs={<key1> : ..., <key2> : ...,}` and similarly for `targets` AND the items returned from the model `outputs={<key1> : ...,}`. HENCE items which reference processing applied to the resultant items after unpacking should drop the first qualifier and just use `<key>`. NOTE: this may reduce the consistency between parts code, will look into remedying this at some point.

