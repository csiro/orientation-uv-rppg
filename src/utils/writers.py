# import h5py
# import torch
# from pathlib import Path
# from abc import ABC, abstractmethod

# from typing import *
# from numpy import ndarray
# from torch import Tensor


# class Writer:
#     """
#     """
#     def __init__(self, path: str, *args, **kwargs) -> None:
#         # super(Writer, self).__init__(*args, **kwargs)
#         self.path = Path(path)
#         self.path.parent.mkdir(parents=True, exist_ok=True)

#     def __call__(self, inputs, targets, sources, outputs, predictions, losses, metrics, *args, **kwargs) -> None:
#         # 
#         data = {
#             # "targets": (targets, "labels", ["interface"]), # can always re-reference at run-time
#             # "predictions": (predictions, "traces", ["interface"]),
#             "metrics": (metrics, None, [])
#         }

#         #
#         with h5py.File(self.path, "a") as fp:
#             for idx, src in enumerate(sources["sample"].data):
#                 # 
#                 if src in fp.keys(): del fp[src]

#                 # Source attributes
#                 grp = fp.create_group(name=src)
#                 for key, val in vars(sources["sample"]).items():
#                     if key not in ["data", "interface", "attrs"]:
#                         grp.attrs.create(key, str(val[idx]))

#                 # Targets, Predictions, Etc.
#                 for dkey, (sdata, skey, excl_keys) in data.items():
#                     grp = fp.create_group(name=f"{src}/{dkey}")
#                     sdata = vars(sdata[skey]) if skey is not None else sdata
#                     for key, val in sdata.items():
#                         key = key.split("/")[-1]
#                         if key in excl_keys: continue
#                         if type(val) in [ndarray, Tensor]:
#                             if type(val) == Tensor: val = val.cpu().numpy()
#                             if len(val.shape) > 1: val = val[idx] # else flat
#                             grp.create_dataset(name=f"{key}", data=val)
#                         else:
#                             if type(val) in [torch.device]: continue
#                             grp.attrs.create(key, str(val[idx]))
