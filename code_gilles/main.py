# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 14:52:11 2018

@author: tgill
"""

import numpy as np
import pandas as pd
#import nltk
from time import time
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing 
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from xgboost import XGBClassifier
#from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from collections import Counter

#node_information = pd.read_csv('node_information.csv', header=None, names=['ID', 'Year', 'Title', 'Authors', 'Journal', 'Abstract'])
#node_information = pd.read_csv('node_information.csv', header=None, names=['ID', 'Year', 'Title', 'Authors', 'Journal', 'Abstract'])
#training_set = pd.read_csv('training_set.txt', header=None, names=['Target', 'Source', 'Edge'], delim_whitespace=True)
#testing_set = pd.read_csv('testing_set.txt', header=None, names=['Target', 'Source'], delim_whitespace=True)

from utils import common, overlap, loop, parallel_loop, overlap_df, tfidf
from wrappers import NeuralNet, NLPNN, Mean
from models import nn, siamois, decomposable_attention, esim, siamois_seq, siamois_cnn, siamois_char

#


#t=time()
#print("Tfidf")
##tfidf_vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
##abstracts_source = train['Abstract_source'].values
##abstracts_target = train['Abstract_target'].values
#abstracts_source_test = test['Abstract_source'].values
#abstracts_target_test = test['Abstract_target'].values
##all_abstracts = np.concatenate((abstracts_source,abstracts_target))
##tfidf_vect.fit(all_abstracts)
#print("tf_idf fitted")
#vect_source_test = tfidf_vect.transform(abstracts_source_test)
#print("source transformed")
#vect_target_test = tfidf_vect.transform(abstracts_target_test)
#print(vect_target_test.shape)
#print("target transformed")
#print(time()-t)
#t=time()
#test['Tfidf_abstracts_(1,2)']=tfidf(vect_source_test, vect_target_test)
#print(time()-t)
#
#train.to_csv('train_char_ngrams.csv', index=False)
#test.to_csv('test_ngram.csv', index=False)
train = pd.read_csv('train_both.csv')
test = pd.read_csv('test_both.csv')

features = ['Overlap_title', 'Common_authors', 'Date_diff', 'Overlap_abstract', 'Tfidf_cosine_abstracts_nolim', 'Tfidf_cosine_titles', 'Tfidf_abstracts_(1,2)',#, 'Tfidf_abstracts_chars_1,4','Tfidf_abstracts_chars_1,5'
         'Target_degree',
       'Target_nh_subgraph_edges', 'Target_nh_subgraph_edges_plus',
       'Source_degree', 'Source_nh_subgraph_edges',
       'Source_nh_subgraph_edges_plus', 'Preferential attachment', 'Target_core', 'Target_clustering', 'Target_pagerank', 'Source_core',
       'Source_clustering', 'Source_pagerank', 'Common_friends',
       'Total_friends', 'Friends_measure', 'Sub_nh_edges', 'Sub_nh_edges_plus',
       'Len_path',
       'Both',
       'Jaccard'
       #'GRU_Siamois',
       #'CNN_Siamois'
       ]#, 'Jaccard']

elite=[ 'Common_authors', 'Date_diff', 'Tfidf_cosine_titles', 'Tfidf_abstracts_(1,2)',#, 'Tfidf_abstracts_chars_1,4','Tfidf_abstracts_chars_1,5'
       'Preferential attachment', 'Target_clustering', 'Target_pagerank',
       'Source_clustering', 'Source_pagerank', 'Common_friends',
       'Friends_measure',
       #'Len_path',
       'Both'
       #'GRU_Siamois',
       #'CNN_Siamois'
       ]
#features = elite


K = 5
np.random.seed(7)
cv = KFold(n_splits = K, shuffle = True, random_state=1)

X = train[features].values
X_test = test[features].values
y = train['Edge'].values
X=preprocessing.scale(X)
X_test=preprocessing.scale(X_test)

lr = LogisticRegression()
nb = GaussianNB()
rf = RandomForestClassifier()
xgb = XGBClassifier(n_estimators=512, max_depth=6, subsample=0.9, scale_pos_weight=0.8366365291081073)
#cat = CatBoostClassifier()
lgbm = LGBMClassifier(n_estimators=512, max_depth=4, reg_lambda=1., subsample=0.9 )
nn = NeuralNet(nn, batch_size=1024, epochs=40, units=512, dropout=0.4 , layers=3)
nnlp = NLPNN(siamois_cnn, epochs=15, batch_size=1024, num_words=15000, maxlen=150, embedding=False, char=False, maxlen_chars=512)
classifier = lgbm#Mean([lgbm, xgb])

#classifier=nnlp
#features_nlp = ['Abstract_target', 'Abstract_source']
#X = train[features_nlp].values
#X_test = test[features_nlp].values

sumf1=0
pred_test=0
scores=[]
feat_prob = np.empty(X.shape[0])
for i, (idx_train, idx_val) in enumerate(cv.split(train)):
    t=time()
    print("Fold ", i )
    X_train = X[idx_train]
    y_train = y[idx_train]
    X_valid = X[idx_val]
    y_valid = y[idx_val]
    classifier.fit(X_train, y_train)#, eval_set=(X_valid, y_valid))
    pred=classifier.predict_proba(X_valid)
    feat_prob[idx_val] = pred[:,1]
    pred = np.argmax(pred, axis=1)
    #print(classifier.coef_)
    #print(Counter(pred))
    #print(classifier.feature_importances_)
    pred_test_fold = classifier.predict_proba(X_test)
    pred_test+=pred_test_fold
    print(Counter(np.argmax(pred_test_fold, axis=1)))
    score=f1_score(pred,y_valid)
    scores.append(score)
    print(score)
    sumf1 +=score
    print(time()-t)
sumf1 = sumf1/K
print("Total score ")
print(sumf1)
print(scores)

a=lgbm.feature_importances_
b=a/np.sum(a)
c= zip(features, b)
print(c)

pred_test = pred_test/K
pred_t = np.argmax(pred_test, axis=1)
pred_test_feat = pred_test[:,1]
#classifier.fit(X, y)
#pred_t = classifier.predict(X_test)

sub = test.copy()
sub['id']=test.index
sub['category'] = pred_t
sub = sub[['id', 'category']]

sub.to_csv('sub.csv', index=False)
