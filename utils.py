import numpy

EXO_TRAINING = 'data/exo_train.csv.gz'
EXO_TEST = 'data/exo_test.csv.gz'


def load_dataset():
    # Load Exoplanet training data.
    print('\nPlease wait...loading training data')
    train = numpy.loadtxt(EXO_TRAINING, delimiter=",", skiprows=1)
    print("Training data loaded")

    # Load Exoplanet testing data.
    print('\nPlease wait...loading testing data')
    test = numpy.loadtxt(EXO_TEST, delimiter=",", skiprows=1)
    print("Testing data loaded\n")
    return train, test
