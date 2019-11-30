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

training_files = [
'/mnt/ea4524be-1f99-458a-8bbf-13ab4dab310b/training-day(01-12)/DrDoS_DNS.csv',
'/mnt/ea4524be-1f99-458a-8bbf-13ab4dab310b/training-day(01-12)/DrDoS_LDAP.csv',
'/mnt/ea4524be-1f99-458a-8bbf-13ab4dab310b/training-day(01-12)/DrDoS_MSSQL.csv',
'/mnt/ea4524be-1f99-458a-8bbf-13ab4dab310b/training-day(01-12)/DrDoS_NetBIOS.csv',
'/mnt/ea4524be-1f99-458a-8bbf-13ab4dab310b/training-day(01-12)/DrDoS_NTP.csv',
'/mnt/ea4524be-1f99-458a-8bbf-13ab4dab310b/training-day(01-12)/DrDoS_SNMP.csv',
'/mnt/ea4524be-1f99-458a-8bbf-13ab4dab310b/training-day(01-12)/DrDoS_SSDP.csv',
'/mnt/ea4524be-1f99-458a-8bbf-13ab4dab310b/training-day(01-12)/DrDoS_UDP.csv',
'/mnt/ea4524be-1f99-458a-8bbf-13ab4dab310b/training-day(01-12)/Syn.csv',
'/mnt/ea4524be-1f99-458a-8bbf-13ab4dab310b/training-day(01-12)/TFTP.csv',
'/mnt/ea4524be-1f99-458a-8bbf-13ab4dab310b/training-day(01-12)/UDPLag.csv'
]


test_files = [
'./dataset/testing/MSSQL.csv',
'./dataset/testing/NetBIOS.csv',
'./dataset/testing/Syn.csv',
'./dataset/testing/UDP.csv',
'./dataset/testing/UDPLag.csv',
'./dataset/testing/Portmap.csv'
]


def concat(file1, file2):
    print("reading file ", file2)
    df = pd.read_csv(file2, skiprows=1, skipinitialspace=True, index_col=False, low_memory=False)
    df.to_csv(file1, mode='a', index=False, header=False,)
    del df


df = pd.read_csv('/mnt/ea4524be-1f99-458a-8bbf-13ab4dab310b/training-day(01-12)/DrDoS_DNS.csv', skipinitialspace=True, low_memory=False)
df.to_csv('./dataset/training-ds.csv', index=False)

for f in training_files:
    print(f)
    concat('./dataset/training-ds.csv', f)
    
df = pd.read_csv('./dataset/training-ds.csv',low_memory=False)


df.columns = (df.columns.str.replace("^ ", "")).str.replace(" $", "")
df['Timestamp'] = df['Timestamp'].apply(lambda x: utils.date_str_to_ms(x))

# create a dataframe with all training data except the target column
df = df.drop(['Flow ID', 'Source IP', 'Destination IP', 'SimillarHTTP'], axis=1)
df['Flow Bytes/s'] = df['Flow Bytes/s'].astype(np.float32)
df['Flow Packets/s'] = df['Flow Packets/s'].astype(np.float32)


Y_train = df[["Label"]]
dummy = pd.get_dummies(Y_train["Label"])
Y_train = Y_train.drop(columns=["Label"])
Y_train = pd.concat([dummy], axis=1)
Y_train["Portmap"] = 0

X_train = df.drop(['Label'], axis=1)

# Drop rows that have NA, NaN or Inf
X_train.dropna(inplace=True)
indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
X_train = X_train[indices_to_keep].astype(np.float32)

# create a dataframe with only the target column

print(Y_train)

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

model.add(Dense(14, activation="softmax"))
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

df = pd.read_csv('./dataset/testing/LDAP.csv', skipinitialspace=True, low_memory=False)
df.to_csv('./dataset/testing-ds.csv', index=False)

for f in test_files:
    print(f)
    concat('./dataset/testing-ds.csv', f)
    
df = pd.read_csv('./dataset/testing-ds.csv')

df.columns = (df.columns.str.replace("^ ", "")).str.replace(" $", "")
df['Timestamp'] = df['Timestamp'].apply(lambda x: utils.date_str_to_ms(x))

# create a dataframe with all training data except the target column
df = df.drop(['Flow ID', 'Source IP', 'Destination IP', 'SimillarHTTP'], axis=1)
df['Flow Bytes/s'] = df['Flow Bytes/s'].astype(np.float32)
df['Flow Packets/s'] = df['Flow Packets/s'].astype(np.float32)

# Drop rows that have NA, NaN or Inf
df.dropna(inplace=True)

# Remove the Label output
X_test = df.drop(['Label'], axis=1)

# create a dataframe with only the target column

Y_test = df[["Label"]]
dummy = pd.get_dummies(Y_test["Label"])
Y_test = Y_test.drop(columns=["Label"])
Y_test = pd.concat([dummy], axis=1)

scores = model.evaluate(X_test, Y_test, verbose=1)
print("\n============= FINAL SCORE ============\n")
print(model.metrics_names)
print("%s: %.4f" % (model.metrics_names[1], scores[1]))

