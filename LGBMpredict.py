import os
print("old working dir",os.getcwd())
os.chdir("C:\\Users\\Marc\\Desktop\\NLP\\NLP")
print("new working dir",os.getcwd())
#%%
import numpy as np 
import pandas as pd 
import json

#from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from skopt import gp_minimize
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
#print("Start optimization with cma")
#fun = Objective_Function(objective)
#res = cma.fmin(fun, [1e-2,0.5,0.5], 1e-1, options={'maxfevals': 10})

#%%
print("Start optimization with gp_minimize")
fun = Objective_Function(objective)
res = gp_minimize(fun, [(1e-4, 1), (0,1), (0,1)], n_calls=10)

#%%
print("Best parameters found : ")
# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = fun.wbest[0]
lgb_params['subsample'] = fun.wbest[1]
lgb_params['colsample_bytree'] = fun.wbest[2] 
lgb_params['silent'] = False
lgb_params['seed'] = 555
lgb_params['subsample_freq'] = 10  

dfg = open("bestParams1.txt",'w')
json.dump(lgb_params,dfg)
dfg.close()
print(lgb_params)

#%%
print("Refit full model")
lgb_model = LGBMClassifier(**lgb_params)    
lgb_model.fit(train, labels)

#%%
print("Import testset")
test = pd.read_csv('../data/test_fusion.csv', index_col=0)
test = test.drop(drops, axis=1)

test = test.fillna(0)
test = test.values

#%%
y_pred = lgb_model.predict_proba(test)[:,1] 
ind_0 = y_pred < 0.5
ind_1 = np.logical_not(ind_0)
y_pred[ind_0] = 0
y_pred[ind_1] = 1

#%%
result = pd.DataFrame()
result['id'] = range(len(y_pred))
result['category'] = y_pred
result.to_csv('Submissions/submit_lgbm.csv', index=False)

