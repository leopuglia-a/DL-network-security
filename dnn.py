import os
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Dense
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.model_selection import KFold
import utils
import datetime
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dropout
from keras.regularizers import l2
from keras.callbacks import CSVLogger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


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


# read in data using pandas
df = pd.read_csv("dataset/full-dataset.csv", low_memory=False)
df.columns = (df.columns.str.replace("^ ", "")).str.replace(" $", "")
df['Timestamp'] = df['Timestamp'].apply(lambda x: utils.date_str_to_ms(x))
df['Label'] = df['Label'].apply(lambda x: utils.to_bin(x))

# create a dataframe with all training data except the target column
df = df.drop(['Flow ID', 'Source IP', 'Destination IP', 'SimillarHTTP'], axis=1)
df['Flow Bytes/s'] = df['Flow Bytes/s'].astype(np.float32)
df['Flow Packets/s'] = df['Flow Packets/s'].astype(np.float32)

# Drop rows that have NA, NaN or Inf
df.dropna(inplace=True)
indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
df = df[indices_to_keep].astype(np.float32)

# Remove the Label output
X_train = df.drop(['Label'], axis=1)

# create a dataframe with only the target column
# Y_train = df['Label']

Y_train = df[["Label"]]
dummy = pd.get_dummies(Y_train["Label"])
Y_train = Y_train.drop(columns=["Label"])
Y_train = pd.concat([dummy], axis=1)


# loggers
csv_logger = CSVLogger('lstm.csv', append=True, separator=';')
f1_scores = []

scaler = preprocessing.StandardScaler()

# 1. Standardize as variáveis de X_train e X_test usando preprocessing.scale: https://scikit-learn.org/stable/modules/preprocessing.html
# É sempre importante fazer as duas separadas pra evitar que o training tenha alguma ação sobre o testing
X_train = scaler.fit_transform(X_train)

# create model
print("\n\n")
print("============ STARTING TRAINING ============")
model = Sequential()

# add model layers
model.add(Dense(2048, input_dim=83, kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01))) # 2. Regularization é uma forma de reduzir os weights resultantes, diminuindo a chance de overfit
model.add(BatchNormalization()) # 3. Batch norm é uma forma de standizar os pesos resultantes da layer pra melhorar a convergência
model.add(Activation('relu'))
model.add(Dropout(0.3)) # 4. Dropout é uma forma de evitar overfitting, adicionei camadas com 30% de dropout na rede inteira

model.add(Dense(1024, kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(512, kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(256, kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(128, kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(64, kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense( QTD_DE_CLASSES, activation="softmax"))
model.summary()

# compile model using mse as a measure of model performance
model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=[f1]
)

# train model
model.fit(
    X_train,
    Y_train,
    batch_size=utils.batch_size,
    verbose=1,
    epochs=utils.epochs,
)

scores = model.evaluate(X_test, Y_test, verbose=1)
print(model.metrics_names)
print("%s: %.4f" % (model.metrics_names[1], scores[1]))


final_score = sum(f1_scores) / 5.0
print("============= FINAL SCORE ============")
print(final_score)
