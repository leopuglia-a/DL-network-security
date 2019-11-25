import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import utils
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU, Dropout
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import TimeseriesGenerator
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import CSVLogger

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


history_length = 50
batch_size = 256
epochs = 10
n_features = 83

# read in data using pandas
df = pd.read_csv("dataset/full-dataset.csv", low_memory=False)
df.columns = (df.columns.str.replace("^ ", "")).str.replace(" $", "")
df['Timestamp'] = df['Timestamp'].apply(lambda x: utils.date_str_to_ms(x))
df['Label'] = df['Label'].apply(lambda x: utils.to_bin(x))

# create a dataframe with all training data except the target column
df = df.drop(['Flow ID', 'Source IP', 'Destination IP', 'SimillarHTTP'], axis=1)
df['Flow Bytes/s'] = df['Flow Bytes/s'].astype(np.float16)
df['Flow Packets/s'] = df['Flow Packets/s'].astype(np.float64)

# Drop rows that have NA, NaN or Inf
df.dropna(inplace=True)
indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
df = df[indices_to_keep].astype(np.float64)

# Remove the Label output
X = df.drop(['Label'], axis=1)

# create a dataframe with only the target column
Y = df['Label']

# KFOLD
seed = 7
np.random.seed(seed)

# Use special splitter for time series
# Explanation: https://i.stack.imgur.com/fXZ6k.png
kf = TimeSeriesSplit(n_splits=5)
kf.get_n_splits(X)

count_kfold = 1
cvscores = []
csv_logger = CSVLogger('lstm.csv', append=True, separator=';')

for train_index, test_index in kf.split(X, Y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Get the numpy array
    X_train = X_train.values
    X_test = X_test.values
    Y_train = Y_train.values
    Y_test = Y_test.values

    # X_train and Y_train out of sync!
    Y_train = np.roll(Y_train, 1, axis = 0)
    Y_test = np.roll(Y_test, 1, axis = 0)
    
    # Convert the data to a 3d array
    generator = TimeseriesGenerator(X_train, Y_train, length = history_length, batch_size = batch_size)

    # create model
    print("kfold: ", count_kfold)
    model = Sequential()

    # add model layers
    model.add(GRU(units=200, dropout=0.2, recurrent_dropout=0.2, input_shape=(history_length, n_features)))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()

    # compile model using mse as a measure of model performance
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[f1])

    # fit_generator works differently from fit: https://stackoverflow.com/questions/43457862/whats-the-difference-between-samples-per-epoch-and-steps-per-epoch-in-fit-g
    # So in order to use the whole dataset we need to calculate the exact number of steps per epoch in order to fully pass the entire dataset each epoch
    steps_epoch = math.floor((len(X_train) - history_length)/batch_size)

    # csv_logger = CSVLogger('lstm.csv', append=True, separator=';')

    # train model
    hist = model.fit_generator(
        generator,
        verbose=1,
        steps_per_epoch=steps_epoch,
        epochs=epochs,
        callbacks=[csv_logger]
    )

    generator = TimeseriesGenerator(X_test, Y_test, length = history_length, batch_size = batch_size)
    scores = model.evaluate_generator(generator, verbose=0)
    print(model.metrics_names)
    print("%s: %.4f" % (model.metrics_names[1], scores[1] * 100))
    count_kfold += 1


plot_model(model, to_file='model.png')

plt.plot(hist.history['f1'])
plt.title('Model f1')
plt.ylabel('f1')
plt.xlabel('Epoch')
plt.legend(['f1'], loc='upper right')
plt.savefig('f1.png')
# plt.show()