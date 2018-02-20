# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 11:56:17 2018

@author: tgill
"""
import pandas as pd
import numpy as np
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from utils import common, overlap, loop, parallel_loop, overlap_df, tfidf
#import igraph as ig
import networkx as nx
from keras.preprocessing import text, sequence

def get_data():
    node_information = pd.read_csv('node_information.csv', header=None, names=['ID', 'Year', 'Title', 'Authors', 'Journal', 'Abstract'])
    node_information = pd.read_csv('node_information.csv', header=None, names=['ID', 'Year', 'Title', 'Authors', 'Journal', 'Abstract'])
    training_set = pd.read_csv('training_set.txt', header=None, names=['Target', 'Source', 'Edge'], delim_whitespace=True)
    #testing_set = pd.read_csv('testing_set.txt', header=None, names=['Target', 'Source'], delim_whitespace=True)

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
    
    print("Merge")
    train = pd.merge(training_set, node_information, how='left', left_on='Target_ID', right_index=True)
    train = pd.merge(train, node_information, how='left', left_on='Source_ID', right_index=True, suffixes=['_target', '_source'])
    #train.to_csv('train_blank.csv', index=False)
    
    #train = pd.read_csv('train_blank.csv')
    #train.to_csv('train.csv', index=False)
    
    t = time()
    print("Add overlapping titles")
    train['Overlap_title'] = train.apply(lambda row :overlap(row, 'Title'), axis=1)
    print("Add common_authors")
    train['Common_authors'] = train.apply(lambda row :common(row, 'Authors'), axis=1)
    print("Add overlapping abstract")
    train['Overlap_abstract'] = train.apply(lambda row :overlap(row, 'Abstract'), axis=1)
    print("Date difference")
    train['Date_diff'] = (train['Year_source']-train['Year_target']).abs()
    print(time()-t)
    
    #train.to_csv('train_basic.csv', index=False)
    #print("Loading set")
    #train = pd.read_csv('train_basic.csv')
    
    #print("Loaded")
    t=time()
    print("Tfidf")
    tfidf_vect = TfidfVectorizer(stop_words="english")
    abstracts_source = train['Abstract_source'].values
    abstracts_target = train['Abstract_target'].values
    all_abstracts = np.concatenate((abstracts_source,abstracts_target))
    tfidf_vect.fit(all_abstracts)
    print("tf_idf fitted")
    vect_source = tfidf_vect.transform(abstracts_source)
    print("source transformed")
    vect_target = tfidf_vect.transform(abstracts_target)
    print("target transformed")
    train['Tfidf_cosine_abstracts_nolim']=tfidf(vect_source, vect_target)
    print(time()-t)
    
    #train.to_csv('train_basic_tfidf.csv', index=False)
    #train = pd.read_csv('train_basic_tfidf.csv')
    
    t=time()
    print("Tfidf")
    tfidf_vect = TfidfVectorizer(stop_words="english")
    titles_source = train['Title_source'].values
    titles_target = train['Title_target'].values
    all_abstracts = np.concatenate((titles_source,titles_target))
    tfidf_vect.fit(all_abstracts)
    print("tf_idf fitted")
    vect_source = tfidf_vect.transform(titles_source)
    print("source transformed")
    vect_target = tfidf_vect.transform(titles_target)
    print("target transformed")
    train['Tfidf_cosine_titles']=tfidf(vect_source, vect_target)
    print(time()-t)
    
    #train.to_csv('train_basic_tfidf_title.csv', index=False)
    #train = pd.read_csv('train_basic_tfidf_title.csv')
    
    
    return train

def get_test():
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
    testing_set['Target_ID']= testing_set.apply(lambda row : ID_pos[row[0]], axis=1)
    testing_set['Source_ID']= testing_set.apply(lambda row : ID_pos[row[1]], axis=1)
    training_set['Target_ID']= training_set.apply(lambda row : ID_pos[row[0]], axis=1)
    training_set['Source_ID']= training_set.apply(lambda row : ID_pos[row[1]], axis=1)
    
    
    print("Merge")
    test = pd.merge(testing_set, node_information, how='left', left_on='Target_ID', right_index=True)
    test = pd.merge(test, node_information, how='left', left_on='Source_ID', right_index=True, suffixes=['_target', '_source'])
    
    train = pd.merge(training_set, node_information, how='left', left_on='Target_ID', right_index=True)
    train = pd.merge(train, node_information, how='left', left_on='Source_ID', right_index=True, suffixes=['_target', '_source'])
    #train.to_csv('train_blank.csv', index=False)
    
    #train = pd.read_csv('train_blank.csv')
    #train.to_csv('train.csv', index=False)
    
    t = time()
    print("Add overlapping titles")
    test['Overlap_title'] = test.apply(lambda row :overlap(row, 'Title'), axis=1)
    print("Add common_authors")
    test['Common_authors'] = test.apply(lambda row :common(row, 'Authors'), axis=1)
    print("Add overlapping abstract")
    test['Overlap_abstract'] = test.apply(lambda row :overlap(row, 'Abstract'), axis=1)
    print("Date difference")
    test['Date_diff'] = (test['Year_source']-test['Year_target']).abs()
    print(time()-t)
    
    #train.to_csv('train_basic.csv', index=False)
    #print("Loading set")
    #train = pd.read_csv('train_basic.csv')
    
    #print("Loaded")
    t=time()
    print("Tfidf asbtract")
    tfidf_vect = TfidfVectorizer(stop_words="english")
    abstracts_source_tr = train['Abstract_source'].values
    abstracts_target_tr = train['Abstract_target'].values
    titles_source_tr = train['Title_source'].values
    titles_target_tr = train['Title_target'].values
    abstracts_source = test['Abstract_source'].values
    abstracts_target = test['Abstract_target'].values
    all_abstracts = np.concatenate((abstracts_source_tr,abstracts_target_tr))
    tfidf_vect.fit(all_abstracts)
    print("tf_idf fitted")
    vect_source = tfidf_vect.transform(abstracts_source)
    print("source transformed")
    vect_target = tfidf_vect.transform(abstracts_target)
    print("target transformed")
    test['Tfidf_cosine_abstracts_nolim']=tfidf(vect_source, vect_target)
    print(time()-t)
    
    #train.to_csv('train_basic_tfidf.csv', index=False)
    #train = pd.read_csv('train_basic_tfidf.csv')
    
    t=time()
    print("Tfidf title")
    tfidf_vect = TfidfVectorizer(stop_words="english")
    titles_source = test['Title_source'].values
    titles_target = test['Title_target'].values
    all_abstracts_ti = np.concatenate((titles_source_tr,titles_target_tr))
    tfidf_vect.fit(all_abstracts_ti)
    print("tf_idf fitted")
    vect_source = tfidf_vect.transform(titles_source)
    print("source transformed")
    vect_target = tfidf_vect.transform(titles_target)
    print("target transformed")
    test['Tfidf_cosine_titles']=tfidf(vect_source, vect_target)
    print(time()-t)
    
    print("Tfidf abstract char 1,3")
    tfidf_vect = TfidfVectorizer(stop_words="english", analyzer='char', ngram_range=(1, 3))
    tfidf_vect.fit(all_abstracts_ti)
    print("tf_idf fitted")
    vect_source = tfidf_vect.transform(abstracts_source)
    print("source transformed")
    vect_target = tfidf_vect.transform(abstracts_target)
    print(vect_target.shape)
    print("target transformed")
    print(time()-t)
    t=time()
    test['Tfidf_abstracts_chars_1,3']=tfidf(vect_source, vect_target)
    print(time()-t)
    
    #train.to_csv('train_basic_tfidf_title.csv', index=False)
    #train = pd.read_csv('train_basic_tfidf_title.csv')
    
    print("Tfidf absract char 1,4")
    tfidf_vect = TfidfVectorizer(stop_words="english", analyzer='char', ngram_range=(1, 4))
    tfidf_vect.fit(all_abstracts_ti)
    print("tf_idf fitted")
    vect_source = tfidf_vect.transform(abstracts_source)
    print("source transformed")
    vect_target = tfidf_vect.transform(abstracts_target)
    print(vect_target.shape)
    print("target transformed")
    print(time()-t)
    t=time()
    test['Tfidf_abstracts_chars_1,4']=tfidf(vect_source, vect_target)
    print(time()-t)
    
    print("Tfidf abstract char 1,5")
    tfidf_vect = TfidfVectorizer(stop_words="english", analyzer='char', ngram_range=(1, 5))
    tfidf_vect.fit(all_abstracts_ti)
    print("tf_idf fitted")
    vect_source = tfidf_vect.transform(abstracts_source)
    print("source transformed")
    vect_target = tfidf_vect.transform(abstracts_target)
    print(vect_target.shape)
    print("target transformed")
    print(time()-t)
    t=time() 
    test['Tfidf_abstracts_chars_1,5']=tfidf(vect_source, vect_target)
    print(time()-t)
    
    return test

def create_graph(X, y):
    graph = nx.Graph()
    edges=[]
    nodes=set()
    for i in range(len(X)):
        source = X[i][0]
        target = X[i][1]
        nodes.add(source)
        nodes.add(target)
        if y[i]==1:
            edges.append((source, target))
    #print(nodes)
    #print(edges)
    graph.add_nodes_from(nodes)
    #print(list(graph.vs))
    graph.add_edges_from(edges)
    return graph

def create_directed_graph(X, y):
    graph = nx.DiGraph()
    edges=[]
    nodes=set()
    for i in range(len(X)):
        source = X[i][0]
        target = X[i][1]
        nodes.add(source)
        nodes.add(target)
        if y[i]==1:
            edges.append((source, target))
    #print(nodes)
    #print(edges)
    graph.add_nodes_from(nodes)
    #print(list(graph.vs))
    graph.add_edges_from(edges)
    return graph

def vertex_degree(graph, v):
    return graph.degree(v)

def count(graph, nodes):
    c=0
    for node in nodes:
        for node_ in nodes:
            if graph.has_edge(node, node_):
                c+=1
    return c

def subgraphs_edge_number(graph, v):
    #neighbors = graph.neighbors(v)
    neighbors = list(graph[v].keys())
    neighbors_plus = neighbors + [v]
    subgraph = graph.subgraph(neighbors)
    subgraph_plus = graph.subgraph(neighbors_plus)
    sub_edge_num = subgraph.number_of_edges()
    #sub_edge_num = subgraph.size()
    sub_edge_num_plus = subgraph_plus.number_of_edges()
    return sub_edge_num, sub_edge_num_plus

def all_vertex(graph, v):
    neighbors = list(graph[v].keys())
    neighbors_plus = neighbors + [v]
    subgraph = graph.subgraph(neighbors)
    subgraph_plus = graph.subgraph(neighbors_plus)
    sub_edge_num = subgraph.number_of_edges()
    sub_edge_num_plus = subgraph_plus.number_of_edges()
    #sub_edge_num = count(graph, neighbors)
    #sub_edge_num_plus = count(graph, neighbors_plus)
    return graph.degree(v), sub_edge_num, sub_edge_num_plus

def common_friends(graph, u, v):
    return len(nx.common_neighbors(graph, u, v))

def total_friends(graph, u, v):
    neighbors_u = list(graph[u].keys())
    neighbors_v = list(graph[v].keys())
    total = list(set(neighbors_u).union(neighbors_v))
    return len(total)

def friends_measure(graph, u, v):
    neighbors_u = list(graph[u].keys())
    neighbors_v = list(graph[v].keys())
    c=0
    for n_u in neighbors_u:
        for n_v in neighbors_v:
            if graph.has_edge(n_u, n_v) or graph.has_edge(n_v, n_u):
                c+=1
                
def subgraph_features(graph, u, v):
    neighbors_u = list(graph[u].keys())
    neighbors_v = list(graph[v].keys())
    neighbors_u_plus = neighbors_u + [u]
    neighbors_v_plus = neighbors_v + [v]
    nh = list(set(neighbors_u).union(neighbors_v))
    nh_plus = list(set(neighbors_u_plus).union(neighbors_v_plus))
    sub_nh = graph.subgraph(nh)
    sub_nh_plus = graph.subgraph(nh_plus)
    return sub_nh.number_of_edges(), sub_nh_plus.number_of_edges()

def shortest_path(graph, u, v):
    return nx.shortest_path_length(graph, u, v)

def all_edges(graph, u, v):
    common_friends = len(list(nx.common_neighbors(graph, u, v)))
    neighbors_u = list(graph[u].keys())
    neighbors_v = list(graph[v].keys())
    nh = list(set(neighbors_u).union(neighbors_v))
    total_friends = len(nh)
    friends_measure=0
    for n_u in neighbors_u:
        for n_v in neighbors_v:
            if graph.has_edge(n_u, n_v) or graph.has_edge(n_v, n_u):
                friends_measure+=1
    neighbors_u_plus = neighbors_u + [u]
    neighbors_v_plus = neighbors_v + [v]
    nh_plus = list(set(neighbors_u_plus).union(neighbors_v_plus))
    sub_nh = graph.subgraph(nh)
    sub_nh_plus = graph.subgraph(nh_plus)
    if not nx.has_path(graph, v, u):
        len_path=-1
    else:
        len_path = nx.shortest_path_length(graph, v, u)
    return common_friends, total_friends, friends_measure, sub_nh.number_of_edges(), sub_nh_plus.number_of_edges(), len_path

def generate_vertex(graph, fs, X, len_fs=3):
    l = X.shape[0]
    feat_target = np.empty((l, len_fs))
    feat_source = np.empty((l, len_fs))
    t1 = time()
    for i, x in enumerate(X):
        t=x[0]
        s=x[1]
        feat_target[i]=fs(graph, t)
        feat_source[i]=fs(graph, s)
        if i%10000==0:
            print(i, l)
            t2=time()
            print(t2-t1)
            t1=t2
    return feat_target, feat_source

def generate_edge(graph, fs, X, len_fs=6):
    l = X.shape[0]
    feat_edge = np.empty((l, len_fs))
    t1 = time()
    for i, x in enumerate(X):
        t=x[0]
        s=x[1]
        feat_edge[i]=fs(graph, t, s)
        if i%10000==0:
            print(i, l)
            t2=time()
            print(t2-t1)
            t1=t2
    return feat_edge

def generate_algo(graph, X):
    res_alloc_index=np.asarray(list(nx.resource_allocation_index(graph, X)))[:,2]
    jac_coef=np.asarray(list(nx.jaccard_coefficient(graph, X)))[:,2]
    ad_adar_idx = np.asarray(list(nx.adamic_adar_index(graph, X)))[:,2]
    pref_att = np.asarray(list(nx.preferential_attachment(graph, X)))[:,2]
    #cn_sound_hop=list(nx.cn_soundarajan_hopcroft(graph, X))
    #ra_sound_hop  =list( nx.ra_index_soundarajan_hopcroft(graph, X))
    #within = list(nx.within_inter_cluster(graph, X))
    return list(res_alloc_index), list(jac_coef), list(ad_adar_idx), list(pref_att)

def generate_numbers(graph, X):
    num_target = np.empty((X.shape[0], 3))
    num_source = np.empty((X.shape[0], 3))
    core_num = nx.core_number(graph)
    clus = nx.clustering(graph)
    page_rank = nx.pagerank(graph)
    for i, x in enumerate(X):
        num_target[i, 0]=core_num[x[0]]
        num_target[i, 1]=clus[x[0]]
        num_target[i, 2]=page_rank[x[0]]
        num_source[i, 0]=core_num[x[1]]
        num_source[i, 1]=clus[x[1]]
        num_source[i, 2]=page_rank[x[1]]
    return num_target, num_source

def all_oriented_vertex(graph, v):
    neighbors_in = graph.predecessors(v)
    neighbors_out = graph.successors(v)
    neighbors = list(set(neighbors_in).union(neighbors_out))
    neighbors_plus = neighbors + [v]
    subgraph = graph.subgraph(neighbors)
    subgraph_plus = graph.subgraph(neighbors_plus)
    scc = nx.number_strongly_connected_components(subgraph)
    wcc = nx.number_weakly_connected_components(subgraph)
    scc_plus = nx.number_strongly_connected_components(subgraph_plus)
    #sub_edge_num = count(graph, neighbors)
    #sub_edge_num_plus = count(graph, neighbors_plus)
    return graph.in_degree(v), graph.out_degree(v), scc, wcc, scc_plus, neighbors_in, neighbors_out, neighbors, neighbors_plus

def generate_oriented(graph, X):
    target_feats=np.empty((X.shape[0], 5))
    source_feats=np.empty((X.shape[0], 5))
    edge_feats = np.empty((X.shape[0], 11))
    l = X.shape[0]
    t1 = time()
    for i, x in enumerate(X):
        t=x[0]
        s=x[1]
        in_d_t, out_d_t, scc_t, wcc_t, sccp_t, n_in_t, n_out_t, n_t, np_t = all_oriented_vertex(graph, t)
        in_d_s, out_d_s, scc_s, wcc_s, sccp_s, n_in_s, n_out_s, n_s, np_s = all_oriented_vertex(graph, s)
        com_in = len(set(n_in_t).intersection(n_in_s))
        com_on = len(set(n_out_t).intersection(n_out_s))
        trans_ts = len(set(n_out_t).intersection(n_in_s))
        trans_st = len(set(n_out_s).intersection(n_in_t))
        friends_measure_st=0
        friends_measure_ts=0
        for ns in n_s:
            for nt in n_t:
                if graph.has_edge(ns, nt):
                    friends_measure_st+=1
                if graph.has_edge(nt, ns):
                    friends_measure_ts+=1
        nh = list(set(n_t).union(n_s))            
        nh_plus = list(set(np_t).union(np_s))
        sub_nh = graph.subgraph(nh)
        sub_nh_plus = graph.subgraph(nh_plus)
        scc = nx.number_strongly_connected_components(sub_nh)
        wcc = nx.number_weakly_connected_components(sub_nh)
        scc_plus = nx.number_strongly_connected_components(sub_nh_plus)
        if not nx.has_path(graph, s, t):
            len_path_st=-1
        else:
            len_path_st = nx.shortest_path_length(graph, s, t)
        if not nx.has_path(graph, t, s):
            len_path_ts=-1
        else:
            len_path_ts = nx.shortest_path_length(graph, t, s)
        target_feats[i]=[in_d_t, out_d_t, scc_t, wcc_t, sccp_t]
        source_feats[i]=[in_d_s, out_d_s, scc_s, wcc_s, sccp_s]
        edge_feats[i]=[com_in, com_on, trans_ts, trans_st, friends_measure_st, friends_measure_ts, scc, wcc, scc_plus, len_path_st, len_path_ts]
        if i%10000==0:
            print(i, l)
            t2=time()
            print(t2-t1)
            t1=t2
    return target_feats, source_feats, edge_feats


        
            

def generate_graph_features(train, K=10):
    t=time()
    X = train[['Target', 'Source']].values
    y = train[['Edge']].values
    target_feats=np.empty((train.shape[0], 5))
    source_feats=np.empty((train.shape[0], 5))
    edge_feats=np.empty((train.shape[0], 11))
    print(edge_feats.shape)
    np.random.seed(7)
    cv = KFold(n_splits = K, shuffle = True, random_state=1)
    for i, (idx_train, idx_val) in enumerate(cv.split(train)):
        t1 = time()
        print(i)
        print(edge_feats[idx_val].shape)
        X_train = X[idx_train]
        y_train = y[idx_train]
        X_valid = X[idx_val]
        y_valid = X[idx_val]
        print("Creating graph")
        graph = create_directed_graph(X_train, y_train)
        print("Generating vertex features")
        feat_target, feat_source, feat_edge = generate_oriented(graph, X_valid)
        target_feats[idx_val] = feat_target
        source_feats[idx_val] = feat_source
        edge_feats[idx_val]=feat_edge
        t2=time()
        print(t2-t1)
        t1=t2
    print(time()-t)
    return target_feats, source_feats, edge_feats

def generate_graph_features_test(train, test):
    t=time()
    X = train[['Target', 'Source']].values
    y = train[['Edge']].values
    X_test = test[['Target', 'Source']].values
    target_feats=np.empty((train.shape[0], 5))
    source_feats=np.empty((train.shape[0], 5))
    edge_feats=np.empty((train.shape[0], 11))
    print(edge_feats.shape)
    t1 = time()
    X_train = X
    y_train = y
    print("Creating graph")
    graph = create_directed_graph(X_train, y_train)
    print("Generating vertex features")
    feat_target, feat_source, feat_edge = generate_oriented(graph, X_test)
    test['Target_indegree'] = feat_target[:,0]
    test['Target_outdegree'] = feat_target[:,1]
    test['Target_scc'] = feat_target[:,2]
    test['Target_wcc'] = feat_target[:,3]
    test['Target_scc_plus'] = feat_target[:,4]
    
    test['Source_indegree'] = feat_source[:,0]
    test['Source_outdegree'] = feat_source[:,1]
    test['Source_scc'] = feat_source[:,2]
    test['Source_wcc'] = feat_source[:,3]
    test['Source_scc_plus'] = feat_source[:,4]
    
    test['Common_in'] = feat_edge[:,0]
    test['Common_out'] = feat_edge[:,1]
    test['Transitive_ts'] = feat_edge[:,2]
    test['Transitive_st'] = feat_edge[:,3]
    test['Friend_measure_st'] = feat_edge[:,4]
    test['Friend_measure_ts'] = feat_edge[:,5]
    test['Scc'] = feat_edge[:,6]
    test['Wcc'] = feat_edge[:,7]
    test['Scc_plus'] = feat_edge[:,8]
    test['Len_path_st'] = feat_edge[:,9]
    test['Len_path_ts'] = feat_edge[:,10]
    t2=time()
    print(t2-t1)
    t1=t2
    print(time()-t)
    return test

def generate_test_features(train, test, K=10, fs=all_edges, len_fs=6):
    t=time()
    X = train[['Target', 'Source']].values
    X_test = test[['Target', 'Source']].values
    y = train[['Edge']].values

    X_train = X
    y_train = y
    print("Creating graph")
    graph = create_graph(X_train, y_train)
    print("Generating vertex features")
    #feat_target, feat_source = generate_vertex(graph, fs, X_valid)
    vertex_target, vertex_source = generate_vertex(graph, all_vertex, X_test)
    print("Generate numbers")
    number_target, number_source = generate_numbers(graph, X_test)
    print("Generate edges")
    feat_edge = generate_edge(graph, all_edges, X_test)
    test['Target_degree'] = vertex_target[:,0] 
    test['Target_nh_subgraph_edges'] = vertex_target[:,1]
    test['Target_nh_subgraph_edges_plus'] = vertex_target[:,2] 
    test['Source_degree'] = vertex_source[:,0] 
    test['Source_nh_subgraph_edges'] = vertex_source[:,1] 
    test['Source_nh_subgraph_edges_plus'] = vertex_source[:,2] 
    test['Preferential attachment'] = test['Target_degree']*test['Source_degree']
    
    test['Target_core'] = number_target[:,0] 
    test['Target_clustering'] = number_target[:,1]
    test['Target_pagerank'] = number_target[:,2] 
    test['Source_core'] = number_source[:,0] 
    test['Source_clustering'] = number_source[:,1]
    test['Source_pagerank'] = number_source[:,2] 
    
    test['Common_friends'] = feat_edge[:,0]
    test['Total_friends'] = feat_edge[:,1]
    test['Friends_measure'] = feat_edge[:,2]
    test['Sub_nh_edges'] = feat_edge[:,3]
    test['Sub_nh_edges_plus'] = feat_edge[:,4]
    test['Len_path'] = feat_edge[:,5]

    return test
        
    
def tokenize(train, test, max_features=15000):
    abstracts_source_tr = train['Abstract_source'].values
    abstracts_target_tr = train['Abstract_target'].values
    abstracts_source = test['Abstract_source'].values
    abstracts_target = test['Abstract_target'].values
    all_abstracts = np.concatenate((abstracts_source_tr,abstracts_target_tr))
    tokenizer = text.Tokenizer(num_words=max_features)
    print('Fit tokenizer')
    tokenizer.fit_on_texts(all_abstracts)
    print('Transform train')
    X_target_tr = tokenizer.texts_to_sequences(abstracts_target_tr)
    X_source_tr = tokenizer.texts_to_sequences(abstracts_source_tr)
    print('Transform test')
    X_target = tokenizer.texts_to_sequences(abstracts_target)
    X_source = tokenizer.texts_to_sequences(abstracts_source)
    train['Token_target_15'] = X_target_tr
    train['Token_source_15'] = X_source_tr
    test['Token_target_15'] = X_target
    test['Token_source_15'] = X_source
    return train, test
    
    

    