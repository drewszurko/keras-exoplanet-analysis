from keras.models import Sequential
from keras import optimizers
from keras import callbacks
from keras.layers import Dense, Dropout
from sklearn.preprocessing import Normalizer

import numpy
import utils


def create_model(units, dropout, lr_rate):
    model = Sequential()
    model.add(Dense(units[0], input_dim=3196, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units[1], activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units[2], activation='sigmoid'))
    adam_optimizer = optimizers.Adam(lr=lr_rate)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    return model


def fit_model(model, patience):
    class_weight = {0: 1, 1: 136}
    cb = callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    model.fit(X_train, y_train, epochs=1000, batch_size=50, callbacks=[cb], class_weight=class_weight,
              validation_data=(X_test, y_test))
    return model


def normalize_data(x_train, x_test):
    normalizer = Normalizer()
    x_train = normalizer.transform(x_train)
    x_test = normalizer.transform(x_test)
    return x_train, x_test


def validate_data(strName, x, y):
    scores = model.evaluate(x, y, batch_size=50, verbose=0)
    print("\n%s %s: %.2f%%\n" % (strName, model.metrics_names[1], scores[1] * 100))


def output_predictions(predictions):
    for i in range(len(predictions))[:10]:
        print('Predicted=%f, Expected=%f' % (round(y_test[i]), round(predictions[i][0])))


if __name__ == '__main__':
    # Config
    dropout = 0.40
    lr_rate = 0.003
    units = [12, 8, 1]
    loss_patience = 1

    # Check that our train/test data is available, then load it.
    train, test = utils.load_dataset()

    # Split train data into input (X) and output (Y) variables.
    X_train = train[:, 1:3197]
    y_train = train[:, 0]

    # Split test data into input (X) and output (Y) variables.
    X_test = numpy.array(test[:, 1:3197])
    y_test = numpy.array(test[:, 0])

    # Normalize train and test features
    X_train, X_test = normalize_data(X_train, X_test)

    # Create model.
    model = create_model(units, dropout, lr_rate)

    # Fit model.
    model = fit_model(model, loss_patience)

    # Evaluate training data on the model.
    validate_data("Train", X_train, y_train)

    # Evaluate test data on the model.
    validate_data("Test", X_test, y_test)

    # Predict our test dataset.
    predictions = model.predict(X_test)

    # Output our test dataset for visualization.
    output_predictions(predictions)
