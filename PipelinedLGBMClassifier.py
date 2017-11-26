import numpy as np 
import pandas as pd 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer

#%%

def gini(actual, pred, cmpcol = 0, sortcol = 1):
    all = np.asarray(np.c_[ actual, pred, np.arange(actual.shape[0]) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
 
    giniSum -= (actual.shape[0] + 1) / 2.
    return giniSum / actual.shape[0]
    
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

gini_score = make_scorer(gini_normalized,greater_is_better=True)
gini_loss  = make_scorer(gini_normalized, greater_is_better=False)



#%%
train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')

#%%

id_test = test['id'].values
target_train = train['target'].values

train = train.drop(['target','id'], axis = 1)
test = test.drop(['id'], axis = 1)

col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(col_to_drop, axis=1)  
test = test.drop(col_to_drop, axis=1)  

train = train.replace(-1, np.nan)
test = test.replace(-1, np.nan)

cat_features = [a for a in train.columns if a.endswith('cat')]

for column in cat_features:
	temp = pd.get_dummies(pd.Series(train[column]))
	train = pd.concat([train,temp],axis=1)
	train = train.drop([column],axis=1)
    
for column in cat_features:
	temp = pd.get_dummies(pd.Series(test[column]))
	test = pd.concat([test,temp],axis=1)
	test = test.drop([column],axis=1)

print(train.values.shape, test.values.shape)

#%%
class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]

                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
                y_pred = clf.predict_proba(X_holdout)[:,1]                

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:,1]
        return res



#%%

# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = 0.02
lgb_params['n_estimators'] = 650
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 10
lgb_params['colsample_bytree'] = 0.8   
lgb_params['min_child_samples'] = 500
lgb_params['feature_fraction'] = 0.9
lgb_params['num_iterations']=900
lgb_params['bagging_freq'] = 1
lgb_params['seed'] = 200

lgb_params2 = {}
lgb_params2['n_estimators'] = 1090
lgb_params2['learning_rate'] = 0.02
lgb_params2['colsample_bytree'] = 0.3   
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 16
lgb_params2['num_iterations']=900
lgb_params2['feature_fraction'] = 0.9
lgb_params2['bagging_freq'] = 1
lgb_params2['seed'] = 200


lgb_params3 = {}
lgb_params3['n_estimators'] = 1100
lgb_params3['max_depth'] = 4
lgb_params3['num_iterations']=900
lgb_params3['learning_rate'] = 0.02
lgb_params3['feature_fraction'] = 0.9
lgb_params3['bagging_freq'] = 1
lgb_params3['seed'] = 200

#%%

lgb_model = LGBMClassifier(**lgb_params)

lgb_model2 = LGBMClassifier(**lgb_params2)

lgb_model3 = LGBMClassifier(**lgb_params3)

#%%
log_model = LogisticRegression()
       
stack = Ensemble(n_splits=6,
        stacker = log_model,
        base_models = (lgb_model))        
        
y_pred = stack.fit_predict(train, target_train, test)        

#%%

pipeline = Pipeline([
    ('classifier', stack)
])
    
pipeline.fit_predict(train, target_train, test) 

pipeline.get_params().keys()


hyperparameters = { 'classifier__learning_rate': [0.02, 0.2],
                    'classifier__n_estimators': [500,1100],
                    'classifier__subsample':[0.7,0.8],
                    'classifier__subsample_freq':[2,10],
                    'classifier__colsample_bytree':[0.3,0.8],
                    'classifier__feature_fraction':[0.9],
                    'classifier__bagging_freq':[1],
                    'classifier__verbose':[1]
                  }
clf = GridSearchCV(pipeline, hyperparameters, cv = 3, scoring = gini_score)
 

#%%

sub_1 = pd.DataFrame()
sub_1['id'] = id_test
sub_1['target'] = y_pred

#%%

sub_1.to_csv('PipeLGBM.csv', index = False)

