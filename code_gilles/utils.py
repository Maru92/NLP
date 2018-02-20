# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:38:36 2018

@author: tgill
"""
import nltk
import numpy as np
import pandas as pd
from multiprocessing import cpu_count, Pool
from sklearn.metrics.pairwise import cosine_similarity

#nltk.download('punkt')
#nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

def overlap(row, field):
    text1 = row[field+'_target']
    text2 = row[field+'_source']
    text1 = stop_words_stems(text1)
    text2 = stop_words_stems(text2)
    overlap = len(set(text1).intersection(set(text2)))
    return overlap

def overlap_df(df, name='Overlap_title', field='Title'):
    df[name] = df.apply(lambda row : overlap(row, field))
    return df


def loop(df, f, field):
    l=[]
    le = len(df)
    for index, row in df.iterrows():
        l.append(f(row, field))
        if index%10000==0:
            print(index, le)
    return f

def parallel_loop(df, f):
    partitions = 2
    data_split = np.array_split(df, partitions)
    pool = Pool(2)
    df = pd.concat(pool.map(f, data_split))
    pool.close()
    pool.join()
    return df
    

def common(row, field='Authors'):
    text1 = row[field+'_target']
    text2 = row[field+'_source']
    if text1!=text1 or text2!=text2:
        return 0
    text1 = text1.split(",")
    text2 = text2.split(",")
    common = len(set(text1).intersection(set(text2)))
    return common
    
def stop_words_stems(txt):
    txt = txt.split(",")
    txt = [token for token in txt if token not in stpwds]
    txt = [stemmer.stem(token) for token in txt]
    return txt    

def tfidf(vect1, vect2):
    l = vect1.shape[0]
    cosine=np.zeros(l)
    for i in range(l):
        cosine[i]=cosine_similarity(vect1[i],vect2[i])
        if i%10000==0:
            print(i, l)
    return cosine
#    return cosine_similarity(vect1, vect2)