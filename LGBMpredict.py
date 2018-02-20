import numpy as np 
import pandas as pd 
import json

from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn import preprocessing 

from skopt import gp_minimize
import cma

#%%
print("Import trainset ...")
features = ['Overlap_title', 'Common_authors', 'Date_diff', 'Overlap_abstract', 
            'Tfidf_cosine_abstracts_nolim', 'Tfidf_cosine_titles', 'Tfidf_abstracts_(1,2)',
         'Target_degree', 'Target_nh_subgraph_edges', 'Target_nh_subgraph_edges_plus',
       'Source_degree', 'Source_nh_subgraph_edges','Source_nh_subgraph_edges_plus', 
       'Preferential attachment', 'Target_core', 'Target_clustering', 'Target_pagerank', 'Source_core',
       'Source_clustering', 'Source_pagerank', 'Common_friends', 'Total_friends', 'Friends_measure', 
       'Sub_nh_edges', 'Sub_nh_edges_plus','Len_path', 'Both',
       'Common_authors_prop','Overlap_journal','WMD_abstract','WMD_title','Common_title_prop',
       'Tfidf_abstract_(1,3)', 'Tfidf_abstract_(1,4)', 'Tfidf_abstract_(1,5)',
         'LGBM_Meta', 'LGBM_Abstract', 'Target_indegree', 'Source_indegree',
        'Target_scc', 'Source_scc', 'Target_wcc', 'Source_wcc', 'Wcc',
       'Len_path_st', 'Len_path_ts'
       ]


train = pd.read_csv('../data/train_directed.csv', index_col=0)

#%%
labels = train['Edge'].values

train = train.fillna(0)
train = train[features].values
train = preprocessing.scale(train)

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
def objective_lgbm(param):
    lgb_params = {}
    lgb_params['learning_rate'] = param[0]
    lgb_params['subsample'] = param[1]
    lgb_params['colsample_bytree'] = param[2] 
    lgb_params['lambda_l1'] = param[3]
    lgb_params['lambda_l2'] = param[4]
    lgb_params['silent'] = True
    lgb_params['seed'] = 555
    lgb_params['subsample_freq'] = 4
    lgb_params['num_iterations'] = 950
    
    lgb_model = LGBMClassifier(**lgb_params)  
    
    K = 5
    cv = KFold(n_splits = K, shuffle = True, random_state=1)
    feat_prob = np.empty(train.shape[0])
    for i, (idx_train, idx_val) in enumerate(cv.split(train)):
        print("Fold ", i )
        X_train = train[idx_train]
        y_train = labels[idx_train]
        X_valid = train[idx_val]
        lgb_model.fit(X_train, y_train)
        
        pred = lgb_model.predict_proba(X_valid)[:,1]
        feat_prob[idx_val] = pred[:,1]
    
    ind_0 = feat_prob < 0.5
    ind_1 = np.logical_not(ind_0)
    y_pred[ind_0] = 0
    y_pred[ind_1] = 1
    return 1-f1_score(labels, y_pred)

#%%
#def PRS(fun, dim, budget, lb, up):
#    eval = 1
#    x = (up - lb)* np.random.random_sample(dim) - lb
#    min_f = fun(x)
#    while eval < budget:
#        x = (up - lb)* np.random.random_sample(dim) - lb
#        feval = fun(x)
#        eval = eval +1 
#        if (feval < min_f):
#            min_f = feval
#
#    return min_f

#%%
#print("Start optimization with cma")
#fun = Objective_Function(objective_lgbm)
#res = cma.fmin(fun, [0.03,0.7,0.74,0.6,0.68], 1e-1, options={'maxfevals': 10})

#%%
print("Start optimization with gp_minimize")
fun = Objective_Function(objective_lgbm)
res = gp_minimize(fun, [(0.01, 0.07), (0.7,1.0), (0.4,1), (0.5,2.0), (0.5,2.0)], n_calls=200)

#%%
#print("Start optimization with PRS")
#fun = Objective_Function(objective)
#PRS(fun, 5, 50, 0, 1)
    
#%%
#lgb_params = {}
#lgb_params['learning_rate'] = 0.02
#lgb_params['num_iterations'] = 10
#lgb_params['subsample'] = 0.8
#lgb_params['subsample_freq'] = 1
#lgb_params['colsample_bytree'] = 0.8
#
#lgb_model = LGBMClassifier(**lgb_params)
#
#pipeline = Pipeline([
#    ('classifier', lgb_model)
#])

#xgb_params = {}
#xgb_params['n_estimators'] = 512
#xgb_params['max_depth'] = 6
#xgb_params['subsample'] = 0.9
#xgb_params['scale_pos_weight'] = 0.8366365291081073
#
#xgb_model = XGBClassifier(**xgb_params)
#
#pipeline = Pipeline([
#    ('classifier', xgb_model)
#])
#    
#hyperparameters_xgb = { 'classifier__learning_rate': sp_uniform(loc=0.0, scale=0.6),  
#                    'classifier__subsample': sp_uniform(loc=0.5, scale=0.5),  
#                    'classifier__colsample_bytree': sp_uniform(loc=0.3, scale=0.7),  
#                    'classifier__colsample_bylevel': sp_uniform(loc=0.3, scale=0.7), 
#                    'classifier__scale_pos_weight': sp_uniform(loc=0.7, scale=0.6),  
#                    'classifier__max_depth': sp_randint(1, 10),   
#                    'classifier__silent': [0],  
#                    'classifier__seed': [555],  
#                    'classifier__n_estimators': sp_randint(500, 1101)  
#                  }
#lambda, alpha

# specify parameters and distributions to sample from
#hyperparameters_lgbm = { 'classifier__learning_rate': sp_uniform(loc=0.0, scale=0.07),
#                    'classifier__num_iterations': sp_randint(800, 1101),
#                    'classifier__subsample': sp_uniform(loc=0.6, scale=0.3),
#                    'classifier__subsample_freq': sp_randint(1, 8),
#                    'classifier__colsample_bytree': sp_uniform(loc=0.3, scale=0.7),
#                    'classifier__lambda_l1': sp_uniform(loc=0.5, scale=0.7),
#                    'classifier__lambda_l2': sp_uniform(loc=0.5, scale=0.7),
#                    'classifier__silent': [False],
#                    'classifier__seed': [555],
#                    'classifier__num_leaves': sp_randint(15, 31),
#                    'classifier__max_bin': sp_randint(125, 255)
#                  }


#best_hyperparameters_lgbm = {'classifier__colsample_bytree': 0.57468972960190079,
#                        'classifier__num_iterations': 871,
#                        'classifier__seed': 555,
#                        'classifier__num_leaves': 13,
#                        'classifier__subsample_freq': 4,
#                        'classifier__max_bin': 151,
#                        'classifier__learning_rate': 0.06799110856925239,
#                        'classifier__subsample': 0.73288363562623562627079,
#                        'classifier__silent': True}


#best_hyperparameters_lgbm_2 = {'silent': False, 
#                          'subsample_freq': 4, 
#                          'learning_rate': 0.024122807580160777, 
#                          'max_bin': 237, 
#                          'subsample': 0.81014788456650577, 
#                          'colsample_bytree': 0.57783657143646716,               
#                          'seed': 555, 
#                          'num_leaves': 25, 
#                          'num_iterations': 960}

# Score: 0.97855
#best_hyperparameters_lgbm_3 = {'silent': True, 
#                          'subsample_freq': 3, 
#                          'learning_rate': 0.033155350111576165, 
#                          'max_bin': 202, 
#                          'subsample': 0.710518376187222, 
#                          'colsample_bytree': 0.7453807492888789,               
#                          'seed': 555, 
#                          'num_leaves': 18, 
#                          'classifier__lambda_l2': 0.6762026513385271, 
#                          'classifier__lambda_l1': 0.5998295958890201,
#                          'num_iterations': 960}

#lgb_model = LGBMClassifier(**best_hyperparameters_2)    
#lgb_model.fit(X_train, y_train)

# run randomized search
#n_iter_search = 70
#clf = RandomizedSearchCV(pipeline, param_distributions=hyperparameters_lgbm,
#                                   n_iter=n_iter_search, cv = 5, scoring='f1')
#
#clf.fit(train, labels)
#
#print("Refiting")
#
##refitting on entire training data using best settings
#clf.refit
#
#bestParam = clf.best_params_
#
#dfg=open("../data/param/bestParams_lgbm_PRS_70.txt",'w')
#json.dump(bestParam,dfg)
#dfg.close()
#
#print(bestParam)

# TODO 
#a = clf.feature_importances_
#print("Features Importance:  ",a/np.sum(a))


##%%
#print("Start optimization with Exhaustive Search")
#
#lgb_params = {}
#lgb_params['learning_rate'] = 0.02
#lgb_params['num_iterations'] = 10
#lgb_params['subsample'] = 0.8
#lgb_params['subsample_freq'] = 1
#lgb_params['colsample_bytree'] = 0.8
#
#lgb_model = LGBMClassifier(**lgb_params)
#
#pipeline = Pipeline([
#    ('classifier', lgb_model)
#])
#    
#hyperparameters = { 'classifier__learning_rate': [0.02, 0.2],
#                    'classifier__num_iterations': [650,1100],
#                    'classifier__subsample': [0.7],
#                    'classifier__subsample_freq': [1,10],
#                    'classifier__colsample_bytree': [0.7,0.9],
#                    'classifier__silent': [False],
#                    'classifier__seed': [200],
#                    'classifier__num_leaves': [16,31],
#                    'classifier__max_depth': [-1, 4],
#                    'classifier__max_bin': [10, 255]
#                  }
#
#
#clf = GridSearchCV(pipeline, hyperparameters, cv = 5, scoring = 'f1')
#
# 
## Fit and tune model
#clf.fit(train, labels)
#
#print("Refiting")
#
##refitting on entire training data using best settings
#clf.refit
#
#bestParam = clf.best_params_
#
#dfg=open("param/bestParams_ES.txt",'w')
#json.dump(bestParam,dfg)
#dfg.close()
#
#print(bestParam)

#%%
print("Best parameters found : ")
# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = fun.wbest[0]
lgb_params['subsample'] = fun.wbest[1]
lgb_params['colsample_bytree'] = fun.wbest[2]
lgb_params['lambda_l1'] = fun.wbest[3]
lgb_params['lambda_l2'] = fun.wbest[4] 
lgb_params['silent'] = False
lgb_params['seed'] = 555
lgb_params['subsample_freq'] = 4
lgb_params['num_iterations'] = 950 

dfg = open("../data/param/bestParams_lgbm_BO_100.txt",'w')
json.dump(lgb_params,dfg)
dfg.close()
print(lgb_params)


#%%
print("Import testset ... ")
test = pd.read_csv('../data/test_directed.csv', index_col=0)

test = test.fillna(0)
test = test[features].values
test = preprocessing.scale(test)


#%%
print("Refit full model with K folds ... ")
lgb_model = LGBMClassifier(**lgb_params)
    
K = 5
cv = KFold(n_splits = K, shuffle = True, random_state=1)
y_pred_prob = np.zeros((test.shape[0],2))
for i, (idx_train, idx_val) in enumerate(cv.split(train)):
    print("Fold ", i )
    X_train = train[idx_train]
    y_train = train[idx_train]
    
    lgb_model.fit(X_train, y_train)
    
    pred_test_fold = lgb_model.predict_proba(test)
    
    y_pred_prob += pred_test_fold
    
y_pred_prob = y_pred_prob/K
y_pred = np.argmax(y_pred_prob, axis=1)

#%%
#print("Prediction ... ")
#y_pred = clf.predict_proba(test)[:,1] 
#ind_0 = y_pred < 0.5
#ind_1 = np.logical_not(ind_0)
#y_pred[ind_0] = 0
#y_pred[ind_1] = 1

#%%
print("Writing ... ")
result = pd.DataFrame()
result['id'] = range(len(y_pred))
result['category'] = y_pred
result = result.astype(int)
result.to_csv('../data/Submissions/submit_lgbm_BO_100.csv', index=False)


