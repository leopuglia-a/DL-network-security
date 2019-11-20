import os
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import KFold
import utils
import datetime
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


batch_size = 128
epochs = 30

# read in data using pandas
df = pd.read_csv("dataset/full-dataset.csv", low_memory=False)
df.columns = (df.columns.str.replace("^ ", "")).str.replace(" $", "")
df['Timestamp'] = df['Timestamp'].apply(lambda x: utils.date_str_to_ms(x))

# create a dataframe with all training data except the target column
X = df.drop(['Label', 'Flow ID', 'Source IP', 'Destination IP', 'SimillarHTTP'], axis=1)

# create a dataframe with only the target column as dummy variables
Y = df['Label'].apply(lambda x: utils.to_bin(x))

# KFOLD
seed = 7
np.random.seed(seed)

kf = KFold(n_splits=5, shuffle=True, random_state=seed)
kf.get_n_splits(X)

count_kfold = 1
f1_scores = []
for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    # create model
    
    print("kfold: ", count_kfold)
    model = Sequential()
    
    # add model layers
    model.add(Dense(2048, activation="relu", input_dim=83))
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
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
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    f1_scores = scores[1]
    count_kfold += 1

final_score = sum(f1_scores) / 5.0
print("============= FINAL SCORE ============")
print(final_score)
