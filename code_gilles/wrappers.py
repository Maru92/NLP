# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 14:08:52 2018

@author: tgill
"""
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils import compute_sample_weight
import numpy as np
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn import preprocessing
import tensorflow as tf
import keras.backend as K
from keras.preprocessing import text, sequence
from time import time
import models


class NeuralNet(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, net, layers=3, units=16, dropout=0.2, class_weight=None, epochs=5, batch_size=256, random_state=None, scale=False, norm=False):
        #super(NeuralNet, self).__init__(units=units, dropout=dropout)
        self.layers = layers
        self.units=units
        self.dropout = dropout
        self.net = net(layers=self.layers, units=self.units, dropout=self.dropout)
        self.net_func = net
        self.class_weight = class_weight
        self.epochs=epochs
        self.batch_size=batch_size
        self.random_state = random_state
        self.scale = scale
        self.norm=norm
        
    def fit(self, X, y=None, sample_weight=None, eval_set=None, verbose=True):
        if verbose:
            verbose=2
        validation_data = eval_set
        n_features = X.shape[1]
        self.classes_ = np.unique(y)
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))
        
            y = np_utils.to_categorical(y)

        self.n_outputs_ = y.shape[1]

        sgd = SGD(lr=0.02, momentum=0.85)#, decay=0.1)

        
        self.net = self.net_func(input_dim = n_features, output_dim=self.n_outputs_, layers=self.layers, units=self.units, dropout=self.dropout)
        if self.random_state is not None:
            np.random.seed(self.random_state+1)
        self.net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])#, gini_normalized])
        if self.random_state is not None:
            np.random.seed(self.random_state+2)
        if self.scale:
            self.scaler = preprocessing.StandardScaler(copy=False)
            X=self.scaler.fit_transform(X)
        if self.norm:
            X=preprocessing.normalize(X)
        if validation_data is not None:
            X_val, y_val = validation_data
            y_val = np_utils.to_categorical(y_val)
            if self.scale:
                X_val = self.scaler.transform(X_val)
            validation_data=(X_val, y_val)
        self.net.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, sample_weight = sample_weight, validation_data=validation_data, verbose=verbose)
        return self
    
    def predict_proba(self, X):
        if self.scale:
            X=self.scaler.transform(X)
        if self.norm:
            X=preprocessing.normalize(X)
        pred =  self.net.predict(X)
        return pred
    
    def predict(self, X):
        if self.scale:
            X=self.scaler.transform(X)
        if self.norm:
            X=preprocessing.normalize(X)
        return np.argmax(self.net.predict(X), axis=1)

        


def precision(y_true, y_pred):
    """Precision metric.
-
-    Only computes a batch-wise average of precision.
-
-    Computes the precision, a metric for multi-label classification of
-    how many selected items are relevant.
-    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

class NLPNN(BaseEstimator, ClassifierMixin, TransformerMixin):
    
    def __init__(self, net, epochs=5, batch_size=256, num_words=10000, maxlen=150, embedding=False, num_chars=90, maxlen_chars=512, char=False, tokenize=True):
        self.net_func=net
        self.epochs=epochs
        self.batch_size=batch_size
        self.num_words=num_words
        self.maxlen=maxlen
        self.tokenizer = text.Tokenizer(num_words=num_words)
        self.embedding=embedding
        self.tokenize = tokenize
        if char:
            self.num_words=num_chars
            self.maxlen=maxlen_chars
            self.tokenizer = text.Tokenizer(num_words=num_chars, char_level=True)
        
    def transform_text(self, X):
        X_target = X[:,0]
        X_source = X[:,1]
        X_target = self.tokenizer.texts_to_sequences(X_target)
        X_source = self.tokenizer.texts_to_sequences(X_source)
        X_target = sequence.pad_sequences(X_target, maxlen=self.maxlen)
        X_source = sequence.pad_sequences(X_source, maxlen=self.maxlen)
        X = [X_target, X_source]
        return X
    
    def pad_text(self, X):
        X_target = X[:,0]
        X_source = X[:,1]
        X_target = sequence.pad_sequences(X_target, maxlen=self.maxlen)
        X_source = sequence.pad_sequences(X_source, maxlen=self.maxlen)
        X = [X_target, X_source]
        return X
        
    def fit(self, X, y, eval_set=None, verbose=True):
        t = time()
        print("Treating_data")
        X_target = X[:,0]
        X_source = X[:,1]
        if self.tokenize:
            X_all = np.concatenate((X_target, X_source))
            print("Training tokenizer")
            self.tokenizer.fit_on_texts(X_all)
            print("Transforming X")
            X = self.transform_text(X)
        else:
            X = self.pad_text(X)
        y = np_utils.to_categorical(y)
        
        self.net = self.net_func(self.maxlen, self.num_words)
        if self.embedding:
            embedding_matrix = self.get_matrix(300)
            self.net = self.net_func(self.maxlen, self.num_words, embedding_matrix=embedding_matrix)
        self.net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        validation_data = eval_set
        if validation_data is not None:
            X_val, y_val = validation_data
            if self.tokenize:
                print("Transforming X_val")
                X_val = self.transform_text(X_val)
            else:
                X_val = self.pad_text(X_val)
            y_val = np_utils.to_categorical(y_val)
            validation_data=(X_val, y_val)
        if verbose:
            verbose=1
        print("Preprocessing", time()-t)
        print("Start training")
        self.net.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, validation_data=validation_data, verbose=verbose)
        return self
    
    def predict_proba(self, X):
        if self.tokenize:
            X = self.transform_text(X)
        else:
            X = self.pad_text(X)
        pred =  self.net.predict(X)
        return pred
    
    def predict(self, X):
        if self.tokenize:
            X = self.transform_text(X)
        else:
            X = self.pad_text(X)
        return np.argmax(self.net.predict(X), axis=1)
    
    def get_matrix(self, embed_size):
        #self.embedding_file = "pretrained/glove.6B/glove.6B.200d.txt"
        self.embedding_file = "pretrained/glove.840B.300d.txt"
        print("Getting emb")
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        #self.embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(self.embedding_file, encoding='utf8'))
        self.embeddings_index={}
        f = open(self.embedding_file, encoding='utf8')
        for line in f:
            values = line.split()
            word = ' '.join(values[:-300])
            coefs = np.asarray(values[-300:], dtype='float32')
            self.embeddings_index[word] = coefs.reshape(-1)
        f.close
        print("DONE")
        all_embs = np.stack(self.embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        print(emb_mean, emb_std)
        self.embed_size = embed_size
        word_index = self.tokenizer.word_index
        nb_words = min(self.num_words, len(word_index))
        #self.embedding_matrix = np.zeros((nb_words, self.embed_size))
        self.embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, self.embed_size))
        tot=0
        nop=0
        for word, i in word_index.items():
            if i >= self.num_words: continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None: 
                self.embedding_matrix[i] = embedding_vector
                nop+=1
            tot+=1
        print(nop, tot)
        return self.embedding_matrix
    
class Mean(_BaseComposition, ClassifierMixin, TransformerMixin):
    
    def __init__(self, estimators, mean='mean', coefs=None):
        self.estimators=estimators
        self.mean = mean
        self.coefs=coefs
        
    def fit(self, X, y):
        for est in self.estimators:
            est.fit(X,y)
        return self
    
    def predict_proba(self, X):
        pred=[]
        for est in self.estimators:
            pred.append(est.predict_proba(X))
        res = np.ones_like(pred[0])
        if self.mean == 'harmonic':
            for p in pred:
                res = res*p
        if self.mean == 'log':
            res = np.exp(np.mean(np.log(pred), axis=0))
        if self.mean=='rank':
            rank=np.zeros_like(pred[0])
            for p in pred:
                prob = p[:,1]
                r1 = prob.argsort().argsort()/len(prob)
                r0=1-r1
                r = np.zeros_like(rank)
                r[:,0]=r0
                r[:,1]=r1
                rank +=r
            res = rank / len(pred)
        if self.coefs is not None:
            for i, p in enumerate(pred):
                res+=self.coefs[i]*p
#            print(res.shape)
#            print(res)
#            print(pred[0])
        if self.mean=='mean':
            for p in pred:
                res +=p
            res = res/len(pred)
        return res
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)