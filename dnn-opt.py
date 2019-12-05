# Eu alterei o codigo pra usar o TensorFlow 2 (com isso, vamos usar o tensorflow.keras)
# O keras virou parte do tensorflow em si

#### Imports ####

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import pandas as pd
import utils
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout, LeakyReLU, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import optimizers
import datatable as dt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# csv_logger = CSVLogger('dnn2.csv', append=True, separator=';')


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
            df = dt.fread(f)
            df.to_csv(file)
        else:
            df = dt.fread(f, skip_to_line=1)
            df.to_csv(file, append=True)

def gen_train_ds(file):
    

    df = dt.fread(file)

    # Sanity check empty entries
    # df.columns = (df.columns.str.replace("^ ", "")).str.replace(" $", "")

    # Remove columns that don't add useful information
    del df[:, 'Unnamed: 0']
    del df[:, 'Flow ID']
    del df[:, 'Source IP']
    del df[:, 'Destination IP']
    del df[:, 'SimillarHTTP']
    del df[:, 'Timestamp']

    df = df.to_pandas()

    # Cast the variables to correct types
    df['Flow Bytes/s'] = df['Flow Bytes/s'].astype(np.float32)
    df['Flow Packets/s'] = df['Flow Packets/s'].astype(np.float32)

    # Drop rows that have NA, NaN or Inf
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    df = df[indices_to_keep]

    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv("./dataset/training-ds.csv")

def gen_test_ds(file):


    df = dt.fread(file)

    # Sanity check empty entries
    # df.columns = (df.columns.str.replace("^ ", "")).str.replace(" $", "")

    # Remove columns that don't add useful information
    # df = df.drop(['Unnamed: 0', 'Flow ID', 'Source IP', 'Destination IP', 'SimillarHTTP', 'Timestamp'], axis=1)
    del df[:, 'Unnamed: 0']
    del df[:, 'Flow ID']
    del df[:, 'Source IP']
    del df[:, 'Destination IP']
    del df[:, 'SimillarHTTP']
    del df[:, 'Timestamp']

    df = df.to_pandas()

    # Cast the variables to correct types
    df['Flow Bytes/s'] = df['Flow Bytes/s'].astype(np.float32)
    df['Flow Packets/s'] = df['Flow Packets/s'].astype(np.float32)

    # Drop rows that have NA, NaN or Inf
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    df = df[indices_to_keep]

    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv("./dataset/testing-ds.csv")

def gen_dataset():
    training = './dataset/training-concat.csv'
    testing = './dataset/testing-concat.csv'

    # concat( training , training_files)
    # concat( testing , testing_files)

    gen_train_ds(training)
    # gen_test_ds(testing)

index = 1

def generate_arrays_from_file(file, batch_size, lb):
    
    while True:
        global index

        df = dt.fread(file, max_nrows=batch_size, skip_to_line=index)


        df = df.to_pandas()
        Y = df.iloc[:, -1]
        X = df.drop(df.columns[[0, -1]], axis=1).astype(np.float64)
        dummy = pd.get_dummies(Y)
        
        Y = pd.concat([dummy], axis=1)
                
        index = index + batch_size
        yield (X.values, Y.values)



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

full_training_files = [
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
testing_files = [
    './dataset/testing/LDAP.csv',
    './dataset/testing/MSSQL.csv',
    './dataset/testing/NetBIOS.csv',
    './dataset/testing/Syn.csv',
]

# csv_logger = CSVLogger('dnn.csv', append=True, separator=';')

# gen_dataset()


TRAIN_CSV = "./dataset/training-ds.csv"
# TEST_CSV = "./dataset/testing-ds.csv"
 
# # initialize the number of epochs to train for and batch size
# NUM_EPOCHS = 5
BS = 256
 
# # initialize the total number of training and testing image
NUM_TRAIN = 0
# NUM_TEST = 0


f = open(TRAIN_CSV, "r")
labels = {'BENIGN', 'DrDoS_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NTP', 'DrDoS_NetBIOS', 'DrDoS_SNMP', 'DrDoS_SSDP', 'DrDoS_UDP', 'Syn', 'TFTP', 'UDP-lag', 'WebDDoS'}
testLabels = ['BENIGN', 'DrDoS_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NTP', 'DrDoS_NetBIOS', 'DrDoS_SNMP', 'DrDoS_SSDP', 'DrDoS_UDP', 'Syn', 'TFTP', 'UDP-lag', 'WebDDoS']
 
# loop over all rows of the CSV file
for line in f:
	# extract the class label, update the labels list, and increment
	# the total number of training images
	NUM_TRAIN += 1
 
# # close the training CSV file and open the testing CSV file
f.close()

# f = open(TEST_CSV, "r")
 
# # loop over the lines in the testing file
# for line in f:
# 	# extract the class label, update the test labels list, and
# 	# increment the total number of testing images
# 	label = line.strip().split(",")[-1]
# 	testLabels.append(label)
# 	NUM_TEST += 1
 
# # close the testing CSV file
# f.close()

# print(NUM_TRAIN)


# # #### UNCOMMENT IF ITS NEEDED TO GENERATE FILES
# # # gen_dataset()

# # #### Read data training ####

# # df = pd.read_csv('./dataset/training-ds.csv',low_memory=False)

# # # Extract the train columns and convert it to a plain numpy array
# # X_train = df.drop(['Label'], axis=1).values

# # # Extract the test column, dummy it and convert it to a plain numpy array
# # Y_train = df[["Label"]]
# # dummy = pd.get_dummies(Y_train["Label"])
# # Y_train = pd.concat([dummy], axis=1)
# # Y_train = Y_train.values


# # scaler = preprocessing.StandardScaler()
# # X_train = scaler.fit_transform(X_train)



lb = LabelBinarizer()
lb.fit(list(labels))
testLabels = lb.transform(testLabels)


trainGen = generate_arrays_from_file(TRAIN_CSV, BS, lb)
# testGen = generate_arrays_from_file(TEST_CSV, BS, lb)

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

opt = optimizers.Adam(learning_rate=0.01)
# opt = optimizers.SGD(lr=0.0001, momentum=0.9, decay=1e-2 / NUM_EPOCHS)

model.compile(
    loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy', precision, recall, f1]
)


history = model.fit_generator(
    trainGen,
    steps_per_epoch= NUM_TRAIN // BS,
    verbose=1,
    epochs=10,
)

# # plt.plot(history.history['accuracy'])
# # plt.plot(history.history['precision'])
# # plt.plot(history.history['recall'])
# # plt.plot(history.history['f1'])
# # plt.title('Model f1')
# # plt.xlabel('Epoch')
# # plt.legend(['accuracy', 'precision', 'recall', 'f1'], loc='lower right')
# # plt.savefig('dnn3.png')



# # #### Testing ####

# # df = pd.read_csv('./dataset/testing-ds.csv',low_memory=False)

# # # Extract the test columns and convert it to a plain numpy array
# # X_test = df.drop(['Label'], axis=1).values

# # # Extract the test column, dummy it and convert it to a plain numpy array
# # Y_test = df[["Label"]]
# # dummy = pd.get_dummies(Y_test["Label"])
# # Y_test = Y_test.drop(columns=["Label"])
# # Y_test = pd.concat([dummy], axis=1)
# # Y_test.columns = ['BENIGN', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NetBIOS', 'Syn']
# # Y_test.insert(1, 'DrDoS_DNS', 0)
# # Y_test.insert(4, 'DrDoS_NTP', 0)
# # Y_test.insert(6, 'DrDoS_SNMP', 0)
# # Y_test.insert(7, 'DrDoS_SSDP', 0)
# # Y_test.insert(8, 'DrDoS_UDP', 0)
# # Y_test.insert(10, 'TFTP', 0)
# # Y_test.insert(11, 'UDP-lag', 0)
# # Y_test.insert(12, 'WebDDoS', 0)
# # Y_test = Y_test.values

# # # Standardize the input variables
# # scaler = preprocessing.StandardScaler()
# # X_test = scaler.fit_transform(X_test)

# # scores = model.evaluate(X_test, Y_test, verbose=0)