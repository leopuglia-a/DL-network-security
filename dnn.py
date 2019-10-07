import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras import backend as K


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


batch_size = 128
epochs = 30

# read in data using pandas
df = pd.read_csv("dataset/final_dataset.csv")


# create a dataframe with all training data except the target column
X = df.drop(columns=["PKT_CLASS"])

# Generating dummy variables
cols = ["PKT_TYPE", "FLAGS", "NODE_NAME_FROM", "NODE_NAME_TO"]

for col in cols:
    dummy = pd.get_dummies(X[col], drop_first=True)
    X = X.drop(columns=[col])
    X = pd.concat([X, dummy], axis=1)


# create a dataframe with only the target column as dummy variables
Y = df[["PKT_CLASS"]]
dummy = pd.get_dummies(Y["PKT_CLASS"])
Y = Y.drop(columns=["PKT_CLASS"])
Y = pd.concat([dummy], axis=1)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# KFOLD
seed = 7
np.random.seed(seed)

kf = KFold(n_splits=5, shuffle=True, random_state=seed)
kf.get_n_splits(X)

count_kfold = 1
cvscores = []

for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    # create model

    print("kfold: ", count_kfold)
    model = Sequential()

    # add model layers
    model.add(Dense(512, activation="relu", input_dim=(127)))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(5, activation="softmax"))
    model.summary()

    # compile model using mse as a measure of model performance
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", f1]
    )

    # train model
    model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        verbose=0,
        epochs=epochs,
    )

    scores = model.evaluate(X_test, Y_test, verbose=0)
    print(model.metrics_names)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))
    count_kfold += 1

print("\n\n\n ========= New Model =========")

count_kfold = 1
cvscores = []

for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    # create model

    print("kfold: ", count_kfold)
    model = Sequential()

    # add model layers
    model.add(Dense(256, activation="relu", input_dim=(127)))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(5, activation="softmax"))
    model.summary()

    # compile model using mse as a measure of model performance
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", f1]
    )

    # train model
    model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        verbose=0,
        epochs=epochs,
    )

    scores = model.evaluate(X_test, Y_test, verbose=0)
    print(model.metrics_names)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))
    count_kfold += 1

count_kfold = 1
cvscores = []

for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    # create model

    print("kfold: ", count_kfold)
    model = Sequential()

    # add model layers
    model.add(Dense(512, activation="relu", input_dim=(127)))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(5, activation="softmax"))
    model.summary()

    # compile model using mse as a measure of model performance
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", f1]
    )

    # train model
    model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        verbose=0,
        epochs=epochs,
    )

    scores = model.evaluate(X_test, Y_test, verbose=0)
    print(model.metrics_names)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))
    count_kfold += 1

count_kfold = 1
cvscores = []

for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    # create model

    print("kfold: ", count_kfold)
    model = Sequential()

    # add model layers
    model.add(Dense(512, activation="relu", input_dim=(127)))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(5, activation="softmax"))
    model.summary()

    # compile model using mse as a measure of model performance
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", f1]
    )

    # train model
    model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        verbose=0,
        epochs=epochs,
    )

    scores = model.evaluate(X_test, Y_test, verbose=0)
    print(model.metrics_names)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))
    count_kfold += 1

count_kfold = 1
cvscores = []

for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    # create model

    print("kfold: ", count_kfold)
    model = Sequential()

    # add model layers
    model.add(Dense(1024, activation="relu", input_dim=(127)))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(5, activation="softmax"))
    model.summary()

    # compile model using mse as a measure of model performance
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", f1]
    )

    # train model
    model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        verbose=0,
        epochs=epochs,
    )

    scores = model.evaluate(X_test, Y_test, verbose=0)
    print(model.metrics_names)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))
    count_kfold += 1
