import os
import tarfile
from six.moves import urllib

# URL path specification
HOUSING_URL = (
    # "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv"
    "https://storage.googleapis.com/mledu-datasets/california_housing_test.csv"
)
HOUSING_DIR = "./raw/"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_DIR):
    # create directory if not already created
    if not os.path.exists(housing_path):
        os.makedirs(housing_path)

    urllib.request.urlretrieve(
        housing_url, housing_path + "california_housing_test.csv"
    )


if __name__ == "__main__":
    fetch_housing_data()
