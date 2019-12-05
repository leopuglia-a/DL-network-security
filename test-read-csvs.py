import numpy as np
import pandas as pd
import datatable as dt
import dask.dataframe as dd

#### Data training merge ####


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

testing_files = [
    './dataset/testing/LDAP.csv',
    './dataset/testing/MSSQL.csv',
    './dataset/testing/NetBIOS.csv',
    './dataset/testing/Syn.csv',
]

def concat(file, list):
    for index, f in enumerate(list):
        print("reading file ", f)
        if index == 0:
            df = dt.fread(f)
            df.to_csv(file)
        else:
            df = dt.fread(f, skip_to_line=1)
            df.to_csv(file, append=True)


def gen_train_ds():
    

    df = dt.fread('/mnt/ea4524be-1f99-458a-8bbf-13ab4dab310b/training-concat.csv', max_nrows=20000000)

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

def gen_test_ds():


    df = dt.fread('/mnt/ea4524be-1f99-458a-8bbf-13ab4dab310b/testing-concat.csv')

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

    df.to_csv("./dataset/testing-ds.csv")


def gen_dataset():
    # concat('./dataset/training-concat.csv', training_files)
    # concat('./dataset/testing-concat.csv', testing_files)

    gen_train_ds()
    gen_test_ds()

gen_dataset()

# #### Read data training ####
# dataf = dt.fread('./dataset/training-ds.csv')

# df = dataf.to_pandas()

# print("%s seconds", (time.time() - start_time))

# #### Read data training ####
# # df = pd.read_csv('./dataset/training-ds.csv',low_memory=False)
# # df = dd.read_csv('./dataset/training/*.csv', low_memory=False)
# # Sanity check empty entries
# # df.columns = (df.columns.str.replace("^ ", "")).str.replace(" $", "")

# print(df.head())

# print(df)
# # Remove columns that don't add useful information
# df = df.drop(['Unnamed: 0', 'Flow ID', 'Source IP', 'Destination IP', 'SimillarHTTP', 'Timestamp'], axis=1)

# # Cast the variables to correct types
# df['Flow Bytes/s'] = df['Flow Bytes/s'].astype(np.float32)
# df['Flow Packets/s'] = df['Flow Packets/s'].astype(np.float32)

# # Drop rows that have NA, NaN or Inf
# df.dropna(inplace=True)
# indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
# df = df[indices_to_keep]

# # Shuffle the dataset
# df = df.sample(frac=1).reset_index(drop=True)

# # Extract the train columns and convert it to a plain numpy array
# X_train = df.drop(['Label'], axis=1).values

# # Extract the test column, dummy it and convert it to a plain numpy array
# Y_train = df[["Label"]]
# dummy = pd.get_dummies(Y_train["Label"])
# Y_train = Y_train.drop(columns=["Label"])
# Y_train = pd.concat([dummy], axis=1)
# Y_train = Y_train.values

# # Standardize the input variables

# print(X_train)