## Some feature inspired from https://www.kaggle.com/jeru666/did-you-think-of-these-features
import pandas as pd
print(pd.__version__)
import numpy as np

#%%
def change_datatype(df):
    int_cols = list(df.select_dtypes(include=['int']).columns)
    for col in int_cols:
        if ((np.max(df[col]) <= 127) and(np.min(df[col] >= -128))):
            df[col] = df[col].astype(np.int8)
        elif ((np.max(df[col]) <= 32767) and(np.min(df[col] >= -32768))):
            df[col] = df[col].astype(np.int16)
        elif ((np.max(df[col]) <= 2147483647) and(np.min(df[col] >= -2147483648))):
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)

def change_datatype_float(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        df[col] = df[col].astype(np.float32)
        
#%% Loading transaction
print("Read ...")
df_transactions = pd.read_csv('../data/transactions.csv') 
df_transactions = pd.concat((df_transactions, pd.read_csv('../data/transactions_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)

change_datatype(df_transactions)
change_datatype_float(df_transactions)

#%% Creating new feature
df_transactions['discount'] = df_transactions['plan_list_price'] - df_transactions['actual_amount_paid']
df_transactions['is_discount'] = df_transactions.discount.apply(lambda x: 1 if x > 0 else 0)
df_transactions['amt_per_day'] = df_transactions['actual_amount_paid'] / df_transactions['payment_plan_days']

date_cols = ['transaction_date', 'membership_expire_date']
for col in date_cols:
    df_transactions[col] = pd.to_datetime(df_transactions[col], infer_datetime_format=True) #format='%Y%m%d'

df_transactions['membership_duration'] = df_transactions.membership_expire_date - df_transactions.transaction_date
df_transactions['membership_duration'] = df_transactions['membership_duration'] / np.timedelta64(1, 'D')
df_transactions['membership_duration'] = df_transactions['membership_duration'].astype(int)

change_datatype(df_transactions)
change_datatype_float(df_transactions)

#%% Import member
print("Read ...")
df_members = pd.read_csv('../data/members_v3.csv')
change_datatype(df_members)
change_datatype_float(df_members)

date_cols = ['registration_init_time']

for col in date_cols:
    df_members[col] = pd.to_datetime(df_members[col], format='%Y%m%d')
    
#--- difference in days ---
df_members['registration_duration'] = pd.to_datetime(20170331,format='%Y%m%d') - df_members.registration_init_time
df_members['registration_duration'] = df_members['registration_duration'] / np.timedelta64(1, 'D')
df_members['registration_duration'] = df_members['registration_duration'].astype(int)

change_datatype(df_members)
change_datatype_float(df_members)

#%% Merge and delete
print("Merge ...")
df_comb = pd.merge(df_transactions, df_members, on='msno', how='inner')
del df_transactions
del df_members

#%%
df_comb['reg_mem_duration'] = df_comb['registration_duration'] - df_comb['membership_duration']
df_comb['autorenew_&_not_cancel'] = ((df_comb.is_auto_renew == 1) == (df_comb.is_cancel == 0)).astype(np.int8)
df_comb['notAutorenew_&_cancel'] = ((df_comb.is_auto_renew == 0) == (df_comb.is_cancel == 1)).astype(np.int8)
df_comb['long_time_user'] = (((df_comb['registration_duration'] / 365).astype(int)) > 1).astype(int)

#Consume memory
datetime_cols = list(df_comb.select_dtypes(include=['datetime64[ns]']).columns)
print("Write ...")
pd.write_csv(df_comb,'../data/trans_mem.csv')