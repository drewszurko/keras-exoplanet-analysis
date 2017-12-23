from __future__ import print_function
from tqdm import tqdm
import requests
import os
import errno
import pandas as pd

# Local/remote location of exoplanet datasets.
EXO_TRAINING = 'data/exo_train.csv.gz'
EXO_TRAINING_URL = 'https://github.com/drewszurko/keras-exoplanet-analysis/raw/master/data/exo_train.csv.gz'

EXO_TEST = 'data/exo_test.csv.gz'
EXO_TEST_URL = 'https://github.com/drewszurko/keras-exoplanet-analysis/raw/master/data/exo_test.csv.gz'


def get_dataset():
    print("\nImporting data... Please wait.")
    # Create 'data' directory if it does not exist.
    dir_availability()

    # Load our training data.
    train = load_dataset("Training", EXO_TRAINING, EXO_TRAINING_URL)

    # Load our testing data.
    test = load_dataset("Testing", EXO_TEST, EXO_TEST_URL)

    return train, test


# Check if 'data' directory exists. Directory must exist for app to function correctly.
def dir_availability():
    try:
        os.makedirs('data')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


# Loads our training and testing data. Try/except will redownload a dataset file if it's is corrupt or missing.
def load_dataset(name, filepath, url):
    try:
        data = pd.read_csv(filepath, delimiter=",", header=0)
        print("%s data imported successfully." % name)
    except Exception:
        print('\n%s data is missing or corrupt. Lets fix this!' % name)
        download_dataset(name, filepath, url)
        data = pd.read_csv(filepath, delimiter=",", header=0)
        print("%s data imported successfully." % name)
    data = data.values
    return data


# Creates http request to download a missing or corrupted dataset.
def download_dataset(name, filepath, url):
    print('\nDownloading %s data.' % name)
    # Create data download request.
    r = requests.get(url, stream=True)

    # Check for successful request.
    if r.status_code != 200:
        return print("\n %s data could not be downloaded. Please try again." % name)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))

    # Writes file to 'data' directory while displaying a download progress bar in the users terminal.
    with open(filepath, 'wb') as f:
        for chunk in tqdm(iterable=r.iter_content(chunk_size=1024), total=int(total_size / 1024), unit='KB', ncols=100):
            f.write(chunk)

    print("Download complete. Importing %s data." % name)
