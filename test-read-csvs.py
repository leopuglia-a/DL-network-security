# import pandas as pd

# df.to_csv('dataset/syn-dataset.csv', mode='a', header=False, index=False)

import pandas as pd
import os 


files = [
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


def concat(file1, file2):
    print("oi")
    df = pd.read_csv(file2, skiprows=1, nrows=10, skipinitialspace=True, index_col=False)
    df.to_csv(file1, mode='a', index=False, header=False,)
    del df


df = pd.read_csv('./dataset/training/DrDoS_DNS.csv', skipinitialspace=True, nrows=10, low_memory=False)
df.to_csv('./dataset/training-ds.csv', index=False)

for f in files:
    print(f)
    concat('./dataset/training-ds/BIGCSV.csv', f)
    
df = pd.read_csv('./dataset/training-ds/BIGCSV.csv')
print(df)
    