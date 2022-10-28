import csv
import os
from collections import namedtuple
from typing import Tuple
from urllib.parse import urlparse
from urllib.request import urlretrieve

dset = namedtuple("dset", ["name", "urls"])


M4_HOURLY = dset(
    "m4_hourly",
    [
        "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/Train/Hourly-train.csv",  # pylint: disable=line-too-long
        "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/Test/Hourly-test.csv",  # pylint: disable=line-too-long
    ],
)

M4_DAILY = dset(
    "m4_daily",
    [
        "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/Train/Daily-train.csv",  # pylint: disable=line-too-long
        "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/Test/Daily-test.csv",  # pylint: disable=line-too-long
    ],
)

M4_WEEKLY = dset(
    "m4_weekly",
    [
        "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/Train/Weekly-train.csv",  # pylint: disable=line-too-long
        "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/Test/Weekly-test.csv",  # pylint: disable=line-too-long
    ],
)

M4_MONTHLY = dset(
    "m4_monthly",
    [
        "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/Train/Monthly-train.csv",  # pylint: disable=line-too-long
        "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/Test/Monthly-test.csv",  # pylint: disable=line-too-long
    ],
)

M4_QUARTERLY = dset(
    "m4_quarterly",
    [
        "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/Train/Quarterly-train.csv",  # pylint: disable=line-too-long
        "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/Test/Quarterly-test.csv",  # pylint: disable=line-too-long
    ],
)

M4_YEARLY = dset(
    "m4_yearly",
    [
        "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/Train/Yearly-train.csv",  # pylint: disable=line-too-long
        "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/Test/Yearly-test.csv",  # pylint: disable=line-too-long
    ],
)

# pylint: disable=too-few-public-methods
class M4Dataset:
    """
    Set of functions to download the different M4 .csv files and parse them
    """

    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".data"))

    @staticmethod
    def _download(dataset):
        for url in dataset.urls:
            file = os.path.basename(urlparse(url).path)
            out_path = os.path.join(M4Dataset.DATA_DIR, file)
            if url.lower().startswith("https"):
                if not os.path.exists(out_path):
                    print(f"Downloading - {url}.")
                    urlretrieve(url, out_path)
                    print("Download complete.")
            else:
                raise ValueError(f"{url} does not start with https")

    @staticmethod
    def _load_m4(
        dataset, train_csv_path: str, test_csv_path: str
    ) -> Tuple[dict, dict]:
        def parse_m4_csv(file_path: str) -> dict:

            result = {}

            with open(file_path, encoding="UTF-8") as f:

                csv_reader = csv.reader(f)

                next(csv_reader)

                for row in csv_reader:

                    key = str(row[0])
                    value = []

                    for v in row[1:]:
                        if v != "":
                            float_value = float(v)
                            value.append(float_value)

                    result[key] = value

            return result

        M4Dataset._download(dataset)

        train_dict = parse_m4_csv(train_csv_path)
        test_dict = parse_m4_csv(test_csv_path)

        return train_dict, test_dict

    @staticmethod
    def _load_m4_hourly(dataset):
        train_csv_path = os.path.join(M4Dataset.DATA_DIR, "Hourly-train.csv")
        test_csv_path = os.path.join(M4Dataset.DATA_DIR, "Hourly-test.csv")
        return M4Dataset._load_m4(dataset, train_csv_path, test_csv_path)

    @staticmethod
    def _load_m4_daily(dataset):
        train_csv_path = os.path.join(M4Dataset.DATA_DIR, "Daily-train.csv")
        test_csv_path = os.path.join(M4Dataset.DATA_DIR, "Daily-test.csv")
        return M4Dataset._load_m4(dataset, train_csv_path, test_csv_path)

    @staticmethod
    def _load_m4_weekly(dataset):
        train_csv_path = os.path.join(M4Dataset.DATA_DIR, "Weekly-train.csv")
        test_csv_path = os.path.join(M4Dataset.DATA_DIR, "Weekly-test.csv")
        return M4Dataset._load_m4(dataset, train_csv_path, test_csv_path)

    @staticmethod
    def _load_m4_monthly(dataset):
        train_csv_path = os.path.join(M4Dataset.DATA_DIR, "Monthly-train.csv")
        test_csv_path = os.path.join(M4Dataset.DATA_DIR, "Monthly-test.csv")
        return M4Dataset._load_m4(dataset, train_csv_path, test_csv_path)

    @staticmethod
    def _load_m4_quarterly(dataset):
        train_csv_path = os.path.join(M4Dataset.DATA_DIR, "Quarterly-train.csv")
        test_csv_path = os.path.join(M4Dataset.DATA_DIR, "Quarterly-test.csv")
        return M4Dataset._load_m4(dataset, train_csv_path, test_csv_path)

    @staticmethod
    def _load_m4_yearly(dataset):
        train_csv_path = os.path.join(M4Dataset.DATA_DIR, "Yearly-train.csv")
        test_csv_path = os.path.join(M4Dataset.DATA_DIR, "Yearly-test.csv")
        return M4Dataset._load_m4(dataset, train_csv_path, test_csv_path)

    @staticmethod
    def _load(dataset):
        if dataset == M4_HOURLY:
            return M4Dataset._load_m4_hourly(dataset)
        if dataset == M4_DAILY:
            return M4Dataset._load_m4_daily(dataset)
        if dataset == M4_WEEKLY:
            return M4Dataset._load_m4_weekly(dataset)
        if dataset == M4_MONTHLY:
            return M4Dataset._load_m4_monthly(dataset)
        if dataset == M4_QUARTERLY:
            return M4Dataset._load_m4_quarterly(dataset)
        if dataset == M4_YEARLY:
            return M4Dataset._load_m4_yearly(dataset)

        raise ValueError(f"Dataset - {dataset.name} not found.")

    @staticmethod
    def load(dataset) -> Tuple[dict, dict]:

        os.makedirs(M4Dataset.DATA_DIR, exist_ok=True)

        train, test = M4Dataset._load(dataset)

        return train, test
