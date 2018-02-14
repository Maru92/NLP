import numpy as np 
import pandas as pd 
import json

from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split

#from skopt import gp_minimize
#import cma

#%%
print("Import trainset ...")
drops = ["Title_target","Authors_target","Journal_target","Abstract_target","Title_source","Authors_source","Journal_source","Abstract_source"]

train = pd.read_csv('../data/train_fusion.csv', index_col=0)
train = train.drop(drops, axis=1)

#%%
labels = train['label']
train = train.drop(['label'], axis=1)

train = train.fillna(0)
train = train.values
labels = labels.values


N_features = train.shape[1]

#%%
print("Splitting CV ...")
X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.25, random_state=55)

#%%
class Objective_Function(object):
# to be able to inerhit from the function class in Python 2, we need (object)
# f and history_f are attributes of the instance
# we have two instances methods __init__ and __call__
# the __call__ allows to then use fun(x)

    def __init__(self, f):
        self.f = f
        self.history_f = []
        self.fbest = np.inf
        self.history_fbest = []
        self.wbest = []
    
    def __call__(self, x):
        """ Calls the actual objective function in f
            and keeps track of the evaluated search
            points within the history_f list.
        """
        
        f = self.f(x)  # evaluate
        self.history_f.append(f)  # store
        if f < self.fbest:
            self.fbest = f
            self.wbest = x
            
        self.history_fbest.append(self.fbest)

        return f

#%%
def objective(param):
    lgb_params = {}
    lgb_params['learning_rate'] = param[0]
    lgb_params['subsample'] = param[1]
    lgb_params['colsample_bytree'] = param[2] 
    lgb_params['silent'] = False
    lgb_params['seed'] = 555
    lgb_params['subsample_freq'] = 10 #Try different
    
    lgb_model = LGBMClassifier(**lgb_params)    
    lgb_model.fit(X_train, y_train)

    #Prediction
    y_pred = lgb_model.predict_proba(X_test)[:,1] 
    ind_0 = y_pred < 0.5
    ind_1 = np.logical_not(ind_0)
    y_pred[ind_0] = 0
    y_pred[ind_1] = 1
    
    return 1-f1_score(y_test, y_pred)

#%%
def PRS(fun, dim, budget, lb, up):
    eval = 1
    x = (up - lb)* np.random.random_sample(dim) - lb
    min_f = fun(x)
    while eval < budget:
        x = (up - lb)* np.random.random_sample(dim) - lb
        feval = fun(x)
        eval = eval +1 
        if (feval < min_f):
            min_f = feval

    return min_f

#%%
#print("Start optimization with cma")
#fun = Objective_Function(objective)
#res = cma.fmin(fun, [1e-2,0.5,0.5], 1e-1, options={'maxfevals': 10})

#%%
#print("Start optimization with gp_minimize")
#fun = Objective_Function(objective)
#res = gp_minimize(fun, [(1e-4, 1), (0,1), (0,1)], n_calls=10)

#%%
#print("Start optimization with PRS")
#fun = Objective_Function(objective)
#PRS(fun, 3, 50, 0, 1)

#%%
print("Start optimization with Exhaustive Search")

lgb_params = {}
lgb_params['learning_rate'] = 0.02
lgb_params['num_iterations'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 1
lgb_params['colsample_bytree'] = 0.8

lgb_model = LGBMClassifier(**lgb_params)

pipeline = Pipeline([
    ('classifier', lgb_model)
])
    
hyperparameters = { 'classifier__learning_rate': [0.02, 0.2],
                    'classifier__num_iterations': [650,1100],
                    'classifier__subsample': [0.7],
                    'classifier__subsample_freq': [1,10],
                    'classifier__colsample_bytree': [0.7,0.9],
                    'classifier__silent': [False],
                    'classifier__seed': [200],
                    'classifier__num_leaves': [16,31],
                    'classifier__max_depth': [-1, 4],
                    'classifier__max_bin': [10, 255]
                  }

scoring = {'F1_score' : make_scorer(f1_score)}

clf = GridSearchCV(pipeline, hyperparameters, cv = 5, scoring = scoring)

 
# Fit and tune model
clf.fit(train, labels)

print("Refiting")

#refitting on entire training data using best settings
clf.refit

bestParam = clf.best_params_

dfg=open("param/bestParams_ES.txt",'w')
json.dump(bestParam,dfg)
dfg.close()

print(bestParam)

#%%
#print("Best parameters found : ")
## LightGBM params
#lgb_params = {}
#lgb_params['learning_rate'] = fun.wbest[0]
#lgb_params['subsample'] = fun.wbest[1]
#lgb_params['colsample_bytree'] = fun.wbest[2] 
#lgb_params['silent'] = False
#lgb_params['seed'] = 555
#lgb_params['subsample_freq'] = 10  
#
#dfg = open("param/bestParams1.txt",'w')
#json.dump(lgb_params,dfg)
#dfg.close()
#print(lgb_params)
#
##%%
#print("Refit full model ... ")
#lgb_model = LGBMClassifier(**lgb_params)    
#lgb_model.fit(train, labels)

#%%
print("Import testset ... ")
test = pd.read_csv('../data/test_fusion.csv', index_col=0)
test = test.drop(drops, axis=1)

test = test.fillna(0)
test = test.values

#%%
print("Prediction ... ")
y_pred = clf.predict_proba(test)[:,1] 
ind_0 = y_pred < 0.5
ind_1 = np.logical_not(ind_0)
y_pred[ind_0] = 0
y_pred[ind_1] = 1

#%%
print("Writing ... ")
result = pd.DataFrame()
result['id'] = range(len(y_pred))
result['category'] = y_pred
result = result.astype(int)
result.to_csv('Submissions/submit_lgbm_es.csv', index=False)

