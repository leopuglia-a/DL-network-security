# Eu alterei o codigo pra usar o TensorFlow 2 (com isso, vamos usar o tensorflow.keras)
# O keras virou parte do tensorflow em si

#### Imports ####

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import pandas as pd
import utils
from sklearn import preprocessing
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import optimizers


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

csv_logger = CSVLogger('dnn2.csv', append=True, separator=';')


#### F1 Function ####
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

def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))


#### Data training merge ####
def concat(file, list):
    for index, f in enumerate(list):
        print("reading file ", f)
        if index == 0:
            df = pd.read_csv(f, skipinitialspace=True, low_memory=False, nrows=100000)
            df.to_csv(file, index=False)
        else:
            df = pd.read_csv(f, skiprows=1, skipinitialspace=True, low_memory=False, nrows=100000)
            df.to_csv(file, mode='a', index=False, header=False)


training_files = [
    './dataset/training/DrDoS_DNS.csv',
    './dataset/training/DrDoS_LDAP.csv',
    './dataset/training/DrDoS_MSSQL.csv',
    './dataset/training/DrDoS_NetBIOS.csv',
    './dataset/training/DrDoS_NTP.csv',
    './dataset/training/DrDoS_SNMP.csv',
    './dataset/training/DrDoS_SSDP.csv',
    './dataset/training/DrDoS_UDP.csv',
    './dataset/training/Syn.csv',
    './dataset/training/TFTP.csv',
    './dataset/training/UDPLag.csv'
]

testing_files = [
    './dataset/testing/LDAP.csv',
    './dataset/testing/MSSQL.csv',
    './dataset/testing/NetBIOS.csv',
    './dataset/testing/Syn.csv',
]

csv_logger = CSVLogger('dnn.csv', append=True, separator=';')

# concat('./dataset/training-ds.csv', training_files)
# concat('./dataset/testing-ds.csv', testing_files)


#### Read data training ####
df = pd.read_csv('./dataset/training-ds.csv',low_memory=False)

# Sanity check empty entries
df.columns = (df.columns.str.replace("^ ", "")).str.replace(" $", "")

# Remove columns that don't add useful information
df = df.drop(['Unnamed: 0', 'Flow ID', 'Source IP', 'Destination IP', 'SimillarHTTP', 'Timestamp'], axis=1)

# # Convert Timestamp column to usable values
# df['Timestamp'] = df['Timestamp'].apply(lambda x: utils.date_str_to_ms(x))

# Cast the variables to correct types
df['Flow Bytes/s'] = df['Flow Bytes/s'].astype(np.float32)
df['Flow Packets/s'] = df['Flow Packets/s'].astype(np.float32)

# Drop rows that have NA, NaN or Inf
df.dropna(inplace=True)
indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
df = df[indices_to_keep]

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Extract the train columns and convert it to a plain numpy array
X_train = df.drop(['Label'], axis=1).values

# Extract the test column, dummy it and convert it to a plain numpy array
Y_train = df[["Label"]]
dummy = pd.get_dummies(Y_train["Label"])
Y_train = Y_train.drop(columns=["Label"])
Y_train = pd.concat([dummy], axis=1)
Y_train = Y_train.values

# Standardize the input variables
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)


# Create the sequential model
print("\n\n")
print("============ STARTING TRAINING ============")
model = Sequential()


model.add(Dense(512, input_dim=81))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.01))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.01))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.01))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.01))

model.add(Dense(13, activation="softmax"))
model.summary()

opt = optimizers.Adam(learning_rate=0.00001)

model.compile(
    loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy', precision, recall, f1]
)

history = model.fit(
    X_train,
    Y_train,
    batch_size=512,
    verbose=1,
    epochs=5,
    callbacks=[csv_logger]
)

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['precision'])
# plt.plot(history.history['recall'])
# plt.plot(history.history['f1'])
# plt.title('Model f1')
# plt.xlabel('Epoch')
# plt.legend(['accuracy', 'precision', 'recall', 'f1'], loc='lower right')
# plt.savefig('dnn3.png')





#### Testing ####
df = pd.read_csv('./dataset/testing-ds.csv',low_memory=False)

# Sanity check empty entries
df.columns = (df.columns.str.replace("^ ", "")).str.replace(" $", "")

# Remove columns that don't add useful information
df = df.drop(['Unnamed: 0', 'Flow ID', 'Source IP', 'Destination IP', 'SimillarHTTP', 'Timestamp'], axis=1)

# # Convert Timestamp column to usable values
# df['Timestamp'] = df['Timestamp'].apply(lambda x: utils.date_str_to_ms(x))

# Cast the variables to correct types
df['Flow Bytes/s'] = df['Flow Bytes/s'].astype(np.float32)
df['Flow Packets/s'] = df['Flow Packets/s'].astype(np.float32)

# Drop rows that have NA, NaN or Inf
df.dropna(inplace=True)
indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
df = df[indices_to_keep]

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Extract the test columns and convert it to a plain numpy array
X_test = df.drop(['Label'], axis=1).values

# Extract the test column, dummy it and convert it to a plain numpy array
Y_test = df[["Label"]]
dummy = pd.get_dummies(Y_test["Label"])
Y_test = Y_test.drop(columns=["Label"])
Y_test = pd.concat([dummy], axis=1)
Y_test.columns = ['BENIGN', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NetBIOS', 'Syn']
Y_test.insert(1, 'DrDoS_DNS', 0)
Y_test.insert(4, 'DrDoS_NTP', 0)
Y_test.insert(6, 'DrDoS_SNMP', 0)
Y_test.insert(7, 'DrDoS_SSDP', 0)
Y_test.insert(8, 'DrDoS_UDP', 0)
Y_test.insert(10, 'TFTP', 0)
Y_test.insert(11, 'UDP-lag', 0)
Y_test.insert(12, 'WebDDoS', 0)
Y_test = Y_test.values

# Standardize the input variables
scaler = preprocessing.StandardScaler()
X_test = scaler.fit_transform(X_test)

scores = model.evaluate(X_test, Y_test, verbose=0)