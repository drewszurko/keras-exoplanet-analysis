from keras.models import Sequential
from keras import optimizers
from keras import callbacks
from keras.layers import Dense, Dropout
from keras.utils import multi_gpu_model
from sklearn.preprocessing import Normalizer
from time import time

import tensorflow as tf
import utils


def create_model(gpus, units, dropout, lr_rate):
    # Create model using all cpu cores.
    with tf.device('/cpu:0'):
        model = Sequential()
        model.add(Dense(units[0], input_dim=3196, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(units[1], activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(units[2], activation='sigmoid'))
        adam_optimizer = optimizers.Adam(lr=lr_rate)

    # If gpu count is not 2-8, return the regular model. Keras auto detects gpu counts 0,1 and 9+ gpus is not supported.
    if gpus not in (2, 9):
        return model
    else:
        model = multi_gpu_model(model, gpus)
        model = model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
        return model


def fit_model(model, patience):
    print("\nTraining model...\n")
    class_weight = {0: 1, 1: 136}
    cb = callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    model.fit(X_train, y_train, epochs=1000, batch_size=50, callbacks=[cb], class_weight=class_weight,
              validation_data=(X_test, y_test))
    return model


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
    scores = model.evaluate(x, y, batch_size=50, verbose=0)
    print("\n%s %s: %.2f%%" % (str_test, model.metrics_names[1], scores[1] * 100))


def print_predictions(predictions, print_results):
    print('\nDisplaying first %s test results:\n' % print_results)
    for i in range(len(predictions))[:print_results]:
        print('Predicted=%.1f, Expected=%.1f' % (round(predictions[i][0]), round(y_test[i])))


if __name__ == '__main__':
    # Config
    dropout = 0.30
    lr_rate = 0.003
    loss_patience = 1
    units = [12, 8, 1]
    # Displays first n test predicted/expected results in the terminal window. Does not affect training/testing.
    print_results = 10
    # Multi gpu support. Replace the below number with your gpu count. Default: gpus=0
    gpus = 0

    # Execution start time, used to calculate total script completion time.
    startTime = time()

    # Check that our train/test data is available, then load it.
    train, test = utils.get_dataset()

    # Split train data into input (X) and output (Y) variables.
    X_train = train[:, 1:3197]
    y_train = train[:, 0]

    # Split test data into input (X) and output (Y) variables.
    X_test = test[:, 1:3197]
    y_test = test[:, 0]

    # Normalize train and test features
    X_train, X_test = normalize_data(X_train, X_test)

    # Create model.
    model = create_model(gpus, units, dropout, lr_rate)

    # Fit model.
    model = fit_model(model, loss_patience)

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
