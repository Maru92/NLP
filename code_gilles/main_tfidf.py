# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:40:20 2018

@author: tgill
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from sklearn.metrics.pairwise import cosine_similarity

def tfidf(vect1, vect2):
    l = vect1.shape[0]
    cosine=np.zeros(l)
    for i in range(l):
        cosine[i]=cosine_similarity(vect1[i],vect2[i])
        if i%10000==0:
            print(i, l)
    return cosine

def get_data():
    node_information = pd.read_csv('node_information.csv', header=None, names=['ID', 'Year', 'Title', 'Authors', 'Journal', 'Abstract'])
    node_information = pd.read_csv('node_information.csv', header=None, names=['ID', 'Year', 'Title', 'Authors', 'Journal', 'Abstract'])
    training_set = pd.read_csv('training_set.txt', header=None, names=['Target', 'Source', 'Edge'], delim_whitespace=True)
    testing_set = pd.read_csv('testing_set.txt', header=None, names=['Target', 'Source'], delim_whitespace=True)

    print("Get valid IDs")
    valid_ids=set()
    for element in training_set.values:
    	valid_ids.add(element[0])
    	valid_ids.add(element[1])
        
    print("Select valid indices from valid IDs")
    index_valid=[i for i, element in enumerate(node_information.values) if element[0] in valid_ids ]
    node_info=node_information.iloc[index_valid]
    
    print("Get index for nodes")
    IDs = []
    ID_pos={}
    for element in node_info.values:
    	ID_pos[element[0]]=len(IDs)
    	IDs.append(element[0])
        
    print("Add ID column for merging")
    training_set['Target_ID']= training_set.apply(lambda row : ID_pos[row[0]], axis=1)
    training_set['Source_ID']= training_set.apply(lambda row : ID_pos[row[1]], axis=1)
    testing_set['Target_ID']= testing_set.apply(lambda row : ID_pos[row[0]], axis=1)
    testing_set['Source_ID']= testing_set.apply(lambda row : ID_pos[row[1]], axis=1)
    
    print("Merge")
    train = pd.merge(training_set, node_information, how='left', left_on='Target_ID', right_index=True)
    train = pd.merge(train, node_information, how='left', left_on='Source_ID', right_index=True, suffixes=['_target', '_source'])
    test = pd.merge(testing_set, node_information, how='left', left_on='Target_ID', right_index=True)
    test = pd.merge(test, node_information, how='left', left_on='Source_ID', right_index=True, suffixes=['_target', '_source'])
    
    #train = train[:1000]
    #test = test[:1000]
    
    #print("Loaded")
    t=time()
    print("Tfidf (1, 3)")
    tfidf_vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
    abstracts_source = train['Abstract_source'].values
    abstracts_target = train['Abstract_target'].values
    all_abstracts = np.concatenate((abstracts_source,abstracts_target))
    tfidf_vect.fit(all_abstracts)
    print("tf_idf fitted")
    print("    Transform train")
    vect_source = tfidf_vect.transform(abstracts_source)
    print("source transformed")
    vect_target = tfidf_vect.transform(abstracts_target)
    print("target transformed")
    train['Tfidf_abstract_(1,3)']=tfidf(vect_source, vect_target)
    print(time()-t)
    t = time()
    print("    Transform test")
    abstracts_source_test = test['Abstract_source'].values
    abstracts_target_test = test['Abstract_target'].values
    vect_source_test = tfidf_vect.transform(abstracts_source_test)
    print("source transformed")
    vect_target_test = tfidf_vect.transform(abstracts_target_test)
    print("target transformed")
    test['Tfidf_abstract_(1,3)']=tfidf(vect_source_test, vect_target_test)
    print(time()-t)
    t = time()
    train.to_csv('train_tfidf_3.csv', index=False)
    test.to_csv('test_tfidf_3.csv', index=False)
    #train = pd.read_csv('train_basic_tfidf.csv')
    
    t=time()
    print("Tfidf (1, 4)")
    tfidf_vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 4))
    abstracts_source = train['Abstract_source'].values
    abstracts_target = train['Abstract_target'].values
    all_abstracts = np.concatenate((abstracts_source,abstracts_target))
    tfidf_vect.fit(all_abstracts)
    print("tf_idf fitted")
    print("    Transform train")
    vect_source = tfidf_vect.transform(abstracts_source)
    print("source transformed")
    vect_target = tfidf_vect.transform(abstracts_target)
    print("target transformed")
    train['Tfidf_abstract_(1,4)']=tfidf(vect_source, vect_target)
    print(time()-t)
    t = time()
    print("    Transform test")
    abstracts_source_test = test['Abstract_source'].values
    abstracts_target_test = test['Abstract_target'].values
    vect_source_test = tfidf_vect.transform(abstracts_source_test)
    print("source transformed")
    vect_target_test = tfidf_vect.transform(abstracts_target_test)
    print("target transformed")
    test['Tfidf_abstract_(1,4)']=tfidf(vect_source_test, vect_target_test)
    print(time()-t)
    t = time()
    train.to_csv('train_tfidf_4.csv', index=False)
    test.to_csv('test_tfidf_4.csv', index=False)
    
    t=time()
    print("Tfidf (1, 5)")
    tfidf_vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 5))
    abstracts_source = train['Abstract_source'].values
    abstracts_target = train['Abstract_target'].values
    all_abstracts = np.concatenate((abstracts_source,abstracts_target))
    tfidf_vect.fit(all_abstracts)
    print("tf_idf fitted")
    print("    Transform train")
    vect_source = tfidf_vect.transform(abstracts_source)
    print("source transformed")
    vect_target = tfidf_vect.transform(abstracts_target)
    print("target transformed")
    train['Tfidf_abstract_(1,5)']=tfidf(vect_source, vect_target)
    print(time()-t)
    t = time()
    print("    Transform test")
    abstracts_source_test = test['Abstract_source'].values
    abstracts_target_test = test['Abstract_target'].values
    vect_source_test = tfidf_vect.transform(abstracts_source_test)
    print("source transformed")
    vect_target_test = tfidf_vect.transform(abstracts_target_test)
    print("target transformed")
    test['Tfidf_abstract_(1,5)']=tfidf(vect_source_test, vect_target_test)
    print(time()-t)
    t = time()
    train.to_csv('train_tfidf_5.csv', index=False)
    test.to_csv('test_tfidf_5.csv', index=False)
    
    return train, test

train, test = get_data()
