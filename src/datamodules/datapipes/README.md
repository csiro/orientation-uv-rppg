# DataPipes
Provide a flexible implementation to dynamically chain function calls based on a configuration file.

## Dataset Pre-processing
### Format
Provides implementations to reformat the data from a defined `DataSource` to a new file structure/format.

### Extract
Provides implementations to perform extraction of information from a defined `DataSource`.

This typically includes spatial localization of the facial region within video frames such as bounding box or landmark detection.

## Dataset Construction
### Sample
Provides implementations to construct a dataset.

- Construct sliced dataset samples.
- Perform dataset sample splitting.
- Perform dataset sample filtering.

## Dataset Processing
### Process
...

### Transform
...

### Augment
Perform stochastic data augmentations to a given `DatasetSample`.
