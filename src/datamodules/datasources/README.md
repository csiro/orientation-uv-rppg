# DataSources
Functionality of this sub-package is as follows:

## Paths
Provide structured methods for accessing associated dataset files. Often this will take the form of a returned `Dict[str, Path]` for each dataset sample iterated over.

> Defines where to access the files.


## Files
Provide structured methods for accessing the data INSIDE the returned dataset files. 

We first define a base method for accessing general file-types e.g. MATLAB or HDF5 where special functionality is required (often not the case).

We then define an inherited class per dataset file which defines dataset-specific methods for reading and minimally formatting the data.

NOTE: All read methods typically implement lazy and sliced reading to minimize I/O overhead.

> Defines how to access the data in the files.


## Models
Models provide an abstraction layer from the underlying dataset files to the specific dataset samples attributes. 

Models should be defined for each file within a set of dataset files, and provide an interface for accessing the file data and returning an interface class e.g. `Video` or `Timeseries`.

> Defines a standard interface for the data.


## DataSources
Top-level packages within the sub-package provide a high level interface to an associated dataset and the contained data.

> Define a dataset-specific interface.
