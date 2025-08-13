from tqdm import tqdm
from src.datamodule.datamodel import DataModel, stack_datamodels
from src.datamodules.datapipes import DataOperation

from typing import *


class StackDataModels(DataOperation):
    def __init__(self, keys: Optional[List[str]] = None, *args, **kwargs) -> None:
        super(StackDataModels, self).__init__(*args, **kwargs)
        self.keys = keys

    def __call__(self, files: Iterable) -> Dict[str, DataModel]:
        # accumulate lists of `DataModel`
        models = {}
        for file in files:
            for sample in file:
                for key, val in sample.items():
                    if key in models:
                        models[key].append(val)
                    else:
                        models[key] = [val]
        
        # stack `DataModel`s
        for key, val in models.items():
            models[key] = stack_datamodels(val)

        return models


class SummaryOfResults(DataOperation):
    def __init__(self, *args, **kwargs) -> None:
        super(SummaryOfResults, self).__init__(*args, **kwargs)

    def __call__(self, model: DataModel) -> None:
        prediction_hr_bpms = model["predictions/signal_processed"].attrs["peak_bpm"]
        target_hr_bpms = model["targets/labels_unnormalized_processed"].attrs["peak_bpm"]

        mae = np.mean(np.abs(prediction_hr_bpms - target_hr_bpms))
        mae_err = np.std(np.abs(prediction_hr_bpms - target_hr_bpms)) / np.sqrt(n_samples)
        print(f"MAE: {mae:.5f} +/- {mae_err:.5f}")

        rmse = np.sqrt(np.mean(np.square(prediction_hr_bpms - target_hr_bpms)))
        rmse_err = np.std(np.square(prediction_hr_bpms - target_hr_bpms)) / np.sqrt(n_samples)
        print(f"RMSE: {rmse:.5f} +/- {rmse_err:.5f}")

        mape = np.mean(np.abs((prediction_hr_bpms - target_hr_bpms) / target_hr_bpms)) * 100
        mape_err = np.std(np.abs((prediction_hr_bpms - target_hr_bpms) / target_hr_bpms)) / np.sqrt(n_samples)
        print(f"MAPE: {mape:.5f} +/- {mape_err:.5f}")

        pcor = np.corrcoef(prediction_hr_bpms, target_hr_bpms)[0][1]
        pcor_err = np.sqrt((1 - pcor**2) / (n_samples - 2))
        print(f"p CORR: {pcor:.5f} +/- {pcor_err:.5f}")

        # snr = np.mean(prediction_snrs)
        # snr_err = np.std(prediction_snrs) / np.sqrt(n_samples)
        # print(f"SNR: {snr:.5f} +/- {snr_err:.5f}")


class PrintMetrics(DataOperation):
    def __init__(self, *args, **kwargs) -> None:
        super(PrintMetrics, self).__init__(*args, **kwargs)

    def __call__(self, metrics: Dict[str, Dict[str, Any]]) -> None:
        for key, val in metrics.items():
            print(key, val)