from __future__ import print_function
from sklearn.preprocessing import Normalizer
from models import build_model, compile_model, fit_model
from time import time
import data_utils


def normalize_data(x_train, x_test):
    normalizer = Normalizer()
    # Fit our normalizer to our training data.
    normalizer.fit(x_train)
    # Transform the training data using our fitted normalizer.
    x_train = normalizer.transform(x_train)
    # Transform the testing data using our x_trained fitted normalizer.
    x_test = normalizer.transform(x_test)
    return x_train, x_test


def validate_data(str_test, x, y):
    scores = model.evaluate(x, y, batch_size=32, verbose=0)
    print("\n%s %s: %.2f%%" % (str_test, model.metrics_names[1], scores[1] * 100))


def print_predictions(predictions, print_results):
    print('\nDisplaying first %s test results:\n' % print_results)
    for i in range(len(predictions))[:print_results]:
        print('Predicted=%.1f, Expected=%.1f' % (round(predictions[i][0]), round(y_test[i])))


if __name__ == '__main__':
    # Execution start time, used to calculate total script runtime.
    startTime = time()

    # Config
    dropout = 0.20
    lr_rate = 0.001
    loss_patience = 1
    units = [32, 16]
    # Displays first n test predicted/expected results in the terminal window. Does not affect training/testing.
    print_results = 10
    # Multi gpu support. Replace the below number with # of gpus. Default: gpus=0
    gpus = 0

    # Check that our train/test data is available, then load it.
    train, test = data_utils.get_dataset()

    # Split train data into input (X) and output (Y) variables.
    X_train = train[:, 1:3197]
    y_train = train[:, 0]

    # Split test data into input (X) and output (Y) variables.
    X_test = test[:, 1:3197]
    y_test = test[:, 0]

    # Normalize train and test features
    X_train, X_test = normalize_data(X_train, X_test)

    # Create model.
    model = build_model(gpus, units, dropout)

    # Compile model.
    model = compile_model(model, lr_rate)

    # Fit model.
    model = fit_model(model, loss_patience, X_train, y_train, X_test, y_test)

    # Evaluate training data on the model.
    validate_data("Train", X_train, y_train)

    # Evaluate test data on the model.
    validate_data("Test", X_test, y_test)

    # Predict our test dataset.
    predictions = model.predict(X_test)

    # Output our test dataset for visualization.
    print_predictions(predictions, print_results)

    # Print script execution time.
    print("\nExecution time: %s %s \n " % (time() - startTime, "seconds"))
