import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model


def fetch_and_normalize_data():
    train = pd.read_csv("./input/digit-recognizer/train.csv")
    y = train['label'].values
    X = train.drop('label', axis=1).values
    X = X / 255

    y = tf.keras.utils.to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    return X_train, X_test, y_train, y_test

def create_model():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), input_shape=(28, 28, 1), padding="same", activation="relu"))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=32, kernel_size=(5, 5), input_shape=(28, 28, 1), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(28, 28, 1), padding="same", activation="relu"))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(28, 28, 1), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(5, 5), input_shape=(28, 28, 1), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization())

    model.add(Dense(10, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])
    return model


def fit_model():
    X_train, X_test, y_train, y_test = fetch_and_normalize_data()
    imageGenerator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15,
                                                                     width_shift_range=0.12,
                                                                     height_shift_range=0.12,
                                                                     shear_range=0.12,
                                                                     zoom_range=0.12,
                                                                     horizontal_flip=False,
                                                                     vertical_flip=False,
                                                                     )

    model = create_model()
    check_point = ModelCheckpoint("best_model.h5", monitor="val_accuracy", verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", patience=3, verbose=1, factor=0.5, min_lr=0.0001)
    history = model.fit_generator(imageGenerator.flow(X_train, y_train, batch_size=64),
                                  epochs=50, validation_data=(X_test, y_test),
                                  verbose=1,
                                  callbacks=[check_point, reduce_lr])
    losses = pd.DataFrame(model.history.history)
    return losses
    

def prediction():
    saved_model = load_model('best_model.h5')
    test = pd.read_csv("./input/digit-recognizer/test.csv")
    test = test.values
    test = test / 255
    test = test.reshape(-1, 28, 28, 1)

    predictions = saved_model.predict_classes(test)
    submission = pd.Series(predictions, name="Label")
    submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), submission], axis=1)
    return submission

if __name__ == "__main__":
    losses = fit_model()
    submission = prediction()
