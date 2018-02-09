import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import json

#%%
train = pd.read_csv('../data/trainset_full.csv', index_col=0)
labels = train['label']
train = train.drop(['label'], axis=1)

train = train.fillna(0)
train = train.values
labels = labels.values

N_features = train.shape[1]

#%% Create the model with Keras Neural Network
# Function to create model, required for KerasClassifier
def create_model(optimizer='rmsprop', init='glorot_uniform'):
	# create model
	model = Sequential()
	model.add(Dense(int(110), input_dim=int(N_features), kernel_initializer=init, activation='relu'))
	model.add(Dense(int(15), kernel_initializer=init, activation='relu'))
	model.add(Dense(int(1), kernel_initializer=init, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])  #The metric f1 isn't available anymore
	return model

# Model to add some parameters (number of nodes, number of hidden layer) in the GridSearchCV
#def create_model_complex(optimizer='rmsprop', init='glorot_uniform', Nnode = 110, Nnodehiddelayer = [15]):
#    model = Sequential()
#    model.add(Dense(Nnode, input_dim=N_feature, kernel_initializer=init, activation='relu'))
#    for n in Nnodehiddelayer:
#       model.add(Dense(n, kernel_initializer=init, activation='relu'))
#    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
#    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#    return model

model = KerasClassifier(build_fn=create_model, verbose=1)

#%% Define th hyperparameters of the GridSearchCV 





optimizers = ['adam'] #, 'rmsprop']
init = ['glorot_uniform'] #, 'normal', 'uniform']
epochs = [10] #, 100, 150]

batches = [1]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)

clf = GridSearchCV(estimator=model, param_grid=param_grid, verbose=1)

#%% Fitting the model and refitting it with the whole dataset and the best parameters encountered 
clf.fit(train, labels)

bestParam = clf.best_params_

print("Best Parameters: ",bestParam)
dfg=open("param/bestParams_NN_mini.txt",'w')
json.dump(bestParam,dfg)
dfg.close()

clf.refit
del train
del labels

#%%
test = pd.read_csv('../data/testset_full.csv', index_col=0)
test = test.fillna(0)
test = test.values

#%% Prediction
predictions_NN = clf.predict(test)

#%%
result = pd.DataFrame()
result['id'] = range(len(predictions_NN))
result['category'] = predictions_NN
result.to_csv('Submissions/submit_nn_mini.csv', index=False)
