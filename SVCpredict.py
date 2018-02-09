import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import json

#%%
train = pd.read_csv('../data/trainset_full.csv', index_col=0)
labels = train['label']
train = train.drop(['label'], axis=1)

train = train.fillna(0)
train = train.values
labels = labels.values

#%% Training
print("Train classifier")

# Set the parameters by cross-validation

 

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3], 'C': [3]}, 

                    {'kernel': ['linear'], 'C': [3]}]

clf = GridSearchCV(SVC(verbose=1), tuned_parameters, cv=5, scoring = 'f1', verbose=1)
clf.fit(train, labels)

bestParam = clf.best_params_

print("Best Parameters: ",bestParam)
dfg=open("param/bestParams_SVC_mini.txt",'w')
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
predictions_SVM = clf.predict(test)

#%%
result = pd.DataFrame()
result['id'] = range(len(predictions_SVM))
result['category'] = predictions_SVM
result.to_csv('Submissions/submit_svc_mini.csv', index=False)
