import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
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
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

batch_size = 32
epochs = 10

# read in data using pandas
df = pd.read_csv('dataset/final_dataset.csv')

#create a dataframe with all training data except the target column
X = df.drop(columns=['PKT_CLASS'])

cols = ['PKT_TYPE','FLAGS','NODE_NAME_FROM', 'NODE_NAME_TO']

for col in cols:
    dummy = pd.get_dummies(X[col], drop_first=True)
    X = X.drop(columns=[col])
    X = pd.concat([X, dummy], axis=1)

print(X.head())

#create a dataframe with only the target column
y = df[['PKT_CLASS']]
dummy = pd.get_dummies(y['PKT_CLASS'])
y = y.drop(columns=['PKT_CLASS'])
y = pd.concat([dummy], axis=1)

print(y.head())

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#create model
model = Sequential()

#get number of columns in training data

#add model layers
model.add(Dense(256, activation='relu', input_dim=(127)))
model.add(Dense(5, activation='softmax'))
model.summary()

#compile model using mse as a measure of model performance
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=[f1])

#train model
with tf.device('/gpu:0'):
    model.fit(X, y, batch_size=batch_size, verbose=1, epochs=epochs, validation_split=0.3)