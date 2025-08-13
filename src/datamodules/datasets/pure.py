from src.datamodules.datasets import Dataset


class PURE_Dataset(Dataset):
    """ Pulse Rate Estimation (PURE) Dataset

    Links:
        DOI:
        Available at: https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure
    """
    name: str = "PURE"

    def __init__(self, *args, **kwargs) -> None:
        super(PURE_Dataset, self).__init__(*args, **kwargs)