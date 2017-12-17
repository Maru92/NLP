def clearall():
    all = [var for var in globals() if var[0] != "_"]
    for var in all:
        del globals()[var]
        print("Var cleared")
        
clearall()

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import json



# fix random seed for reproducibility
seed = 2015
np.random.seed(seed)
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

#%% Loading 
print("Loading 1 ...")
train = pd.read_csv('../data/train.csv')
train = pd.concat((train, pd.read_csv('../data/train_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)
print("Nb obs total", train.shape[0])

train = train.drop_duplicates(['msno'])
print("Nb obs unique II: ", train.shape[0])
y = train['is_churn'].values



print(train.dtypes)

#%% Merge trans_mem
print("Loading 2 ...")
transmem = pd.read_csv('../data/trans_mem_unscaled_categorical.csv')
print("Before transmem : ", transmem.shape[0])
transmem = transmem.drop_duplicates(['msno'])
print("After drop duplicate: ",transmem.shape[0])

print('Categorical encoding')
cat_features = ['payment_method_id','gender','city','registered_via']
for column in cat_features:
	temp = pd.get_dummies(pd.Series(transmem[column]))
	transmem = pd.concat([transmem,temp],axis=1)
	transmem = transmem.drop([column],axis=1)

print('Scaling')
col_to_scale = ['trans_count','long_time_user','reg_mem_duration','registration_duration','membership_duration','discount','amt_per_day','bd','payment_plan_days','plan_list_price','actual_amount_paid']
for c in col_to_scale:
    #print("Column ",c," has ",sum(np.isnan(df_comb[c]))," nan values sur ",df_comb.shape[0]," !")
    moy = np.nanmean(transmem[c])
    transmem[c] = (transmem[c] - moy)/np.sqrt(np.nansum(np.square(transmem[c] - moy)))    

change_datatype(transmem)
change_datatype_float(transmem)

train = pd.merge(train, transmem, how='left', on='msno')
print("end merge")
del transmem

#%% Merge user_FE
print("Loading 3 ...")
userFE = pd.read_csv('../data/user_FE_scaled.csv',dtype={'num_985':np.float32,'num_985.1':np.float32,
                                                              'num_985.2':np.float32,'num_985.3':np.float32, 'num_50':np.float32,'num_50.1':np.float32,
                                                             'num_50.2':np.float32,'num_50.3':np.float32})

print("Before userFE : ", userFE.shape[0])
userFE = userFE.drop_duplicates(['msno'])
print("After drop duplicate: ",userFE.shape[0])
train = pd.merge(train, userFE, how='left', on='msno')
del userFE

#%% Create data & label
print("Nb observation before:", train.shape[0])
train = train.drop_duplicates(['msno'])
print("NB obs after:", train.shape[0])

X = train.drop(['is_churn','msno','msno.1'], axis=1)
print("end drop")
del train

print(X.dtypes)

change_datatype(X)
change_datatype_float(X)

X = X.fillna(0)



X = X.values

N_feature = X.shape[1]
print("Number of features: ",N_feature)
print("Number of observations: ",X.shape[0] )
#%%


#%%
# We could use CV to improve the result
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2015)

#%%
# Function to create model, required for KerasClassifier
#def create_model(optimizer='rmsprop', init='glorot_uniform'):
#	# create model
#	model = Sequential()
#	model.add(Dense(int(110), input_dim=int(N_feature), kernel_initializer=init, activation='relu'))
#	model.add(Dense(int(15), kernel_initializer=init, activation='relu'))
#	model.add(Dense(int(1), kernel_initializer=init, activation='sigmoid'))
#	# Compile model
#	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#	return model

#def create_model_complex(optimizer='rmsprop', init='glorot_uniform', Nnode = 110, Nnodehiddelayer = [15]):
#    model = Sequential()
#    model.add(Dense(Nnode, input_dim=N_feature, kernel_initializer=init, activation='relu'))
#    for n in Nnodehiddelayer:
#       model.add(Dense(n, kernel_initializer=init, activation='relu'))
#    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
#    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#    return model
model = Sequential()
model.add(Dense(int(110), input_dim=int(N_feature), kernel_initializer='glorot_uniform', activation='relu'))
model.add(Dense(int(15), kernel_initializer='glorot_uniform', activation='relu'))
model.add(Dense(int(1), kernel_initializer='glorot_uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'], epochs=1, batch_size=1)

model.fit(X,y)
#model = KerasClassifier(build_fn=create_model, verbose=1)

#%% grid search epochs, batch size and optimizer
#optimizers = ['adam']#,'rmsprop']
#init = ['glorot_uniform']#, 'normal', 'uniform']
#epochs = [1]#, 100, 150]
#batches = [1] #[5, 10, 20]
#param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
#
#clf = GridSearchCV(estimator=model, param_grid=param_grid)

#%% Fitting
#clf.fit(X, y)
#print("End fitting")
#clf.refit
#del X
#del y
#
##%% Export best param
#bestParam = clf.best_params_
#
#dfg=open("bestParams1.txt",'w')
#json.dump(bestParam,dfg)
#dfg.close()
#
#print(bestParam)

#%%
print("Loading 1 ...")
test = pd.read_csv('../data/sample_submission_v2.csv')
test = test.drop_duplicates(['msno'])
testIndex = test['msno']
#%%
print("Loading 2 ...")
transmem = pd.read_csv('../data/trans_mem_unscaled_categorical.csv')
transmem = transmem.drop_duplicates(['msno'])
print('Categorical encoding')
cat_features = ['payment_method_id','gender','city','registered_via']
for column in cat_features:
	temp = pd.get_dummies(pd.Series(transmem[column]))
	transmem = pd.concat([transmem,temp],axis=1)
	transmem = transmem.drop([column],axis=1)

print('Scaling')
col_to_scale = ['trans_count','long_time_user','reg_mem_duration','registration_duration','membership_duration','discount','amt_per_day','bd','payment_plan_days','plan_list_price','actual_amount_paid']
for c in col_to_scale:
    #print("Column ",c," has ",sum(np.isnan(df_comb[c]))," nan values sur ",df_comb.shape[0]," !")
    moy = np.nanmean(transmem[c])
    transmem[c] = (transmem[c] - moy)/np.sqrt(np.nansum(np.square(transmem[c] - moy)))    


change_datatype(transmem)
change_datatype_float(transmem)
 
test = pd.merge(test, transmem, how='left', on='msno')
del transmem
#%%
print("Loading 3 ...")
userFE = pd.read_csv('../data/user_FE_scaled.csv')
userFE = userFE.drop_duplicates(['msno'])   
test = pd.merge(test, userFE, how='left', on='msno')
del userFE

result = pd.DataFrame()
result['msno'] = testIndex

test = test.drop_duplicates(['msno'])
test = test.drop(['msno','is_churn','msno.1'], axis=1)
test = test.fillna(0)

change_datatype(test)
change_datatype_float(test)

test = test.values

#pred = clf.predict(test)
pred = model.predict(test)
del test

#%% Write results

result['is_churn'] = pred
result.to_csv('NN_FE__test.csv', index=False)

