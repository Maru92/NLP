import random
import numpy as np
import pandas as pd
import igraph
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
import nltk
import csv

from gensim.models.word2vec import Word2Vec
path_to_google_news = "../"
import re
import string

import os
print("old working dir",os.getcwd())
os.chdir('c:\\Users\Marc\desktop\NLP\Kaggle')
print("new working dir",os.getcwd())

#%%
punct = string.punctuation.replace('-', '').replace("'",'')
my_regex = re.compile(r"(\b[-']\b)|[\W_]")

def clean_string(string, punct=punct, my_regex=my_regex, to_lower=False):
    if to_lower:
        string = string.lower()
    # remove formatting
    str = re.sub('\s+', ' ', string)
     # remove punctuation
    str = ''.join(l for l in str if l not in punct)
    # remove dashes that are not intra-word
    str = my_regex.sub(lambda x: (x.group(1) if x.group(1) else ' '), str)
    # strip extra white space
    str = re.sub(' +',' ',str)
    # strip leading and trailing white space
    str = str.strip()
    return str


#%% NLTK initialization
nltk.download('punkt') # for tokenization
nltk.download('stopwords')

#stpwds = set(nltk.corpus.stopwords.words("english"))
# Other Approach of stop word to test
with open('smart_stopwords.txt', 'r') as my_file: 
    stpwds = my_file.read().splitlines()
stemmer = nltk.stem.PorterStemmer()
#stemmer = ntlk.stem.SnowballStemmer()
#lemma = ntlk..wordnet.WordNetLemmatizer()
#lemma.lemmatize(string)
#%% data loading and preprocessing 

# the columns of the data frame below are: 
# (1) paper unique ID (integer)
# (2) publication year (integer)
# (3) paper title (string)
# (4) authors (strings separated by ,)
# (5) name of journal (optional) (string)
# (6) abstract (string) - lowercased, free of punctuation except intra-word dashes


with open("testing_set.txt", "r") as f:
    reader = csv.reader(f)
    testing_set  = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]

with open("training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set  = list(reader)

training_set = [element[0].split(" ") for element in training_set]

with open("node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info  = list(reader)

IDs = [element[0] for element in node_info]

#%% TF-IDF
# compute TFIDF vector of each paper
corpus = [element[5] for element in node_info]
vectorizer = TfidfVectorizer(stop_words="english")
# each row is a node in the order of node_info
features_TFIDF = vectorizer.fit_transform(corpus)

#%% Igraph
print("Constructing the igraph")
# the following shows how to construct a graph with igraph
# even though in this baseline we don't use it
# look at http://igraph.org/python/doc/igraph.Graph-class.html for feature ideas

edges = [(element[0],element[1]) for element in training_set if element[2]=="1"]

# some nodes may not be connected to any other node
# hence the need to create the nodes of the graph from node_info.csv,
# not just from the edge list

nodes = IDs

# create empty directed graph
g = igraph.Graph(directed=True)
 
# add vertices
g.add_vertices(nodes)
 
# add edges
g.add_edges(edges)


#%% Cleaned docs
print("Cleaning the docs")
cleaned_docs_abstract = []
for idx, doc in enumerate(corpus):
    # clean
    doc = clean_string(doc, punct, my_regex, to_lower=True)
    # tokenize (split based on whitespace)
    tokens = doc.split(' ')
    # remove stopwords
    tokens = [token for token in tokens if token not in stpwds]
    # remove digits
    tokens = [''.join([elt for elt in token if not elt.isdigit()]) for token in tokens]
    # remove tokens shorter than 3 characters in size
    tokens = [token for token in tokens if len(token)>2]
    # remove tokens exceeding 25 characters in size
    tokens = [token for token in tokens if len(token)<=25]
    cleaned_docs_abstract.append(tokens)
    if idx % round(len(corpus)/10) == 0:
        print(idx)

corpus_title = [element[2] for element in node_info]
cleaned_docs_title = []
for idx, doc in enumerate(corpus_title):
    # clean
    doc = clean_string(doc, punct, my_regex, to_lower=True)
    # tokenize (split based on whitespace)
    tokens = doc.split(' ')
    # remove stopwords
    tokens = [token for token in tokens if token not in stpwds]
    # remove digits
    tokens = [''.join([elt for elt in token if not elt.isdigit()]) for token in tokens]
    # remove tokens shorter than 3 characters in size
    tokens = [token for token in tokens if len(token)>2]
    # remove tokens exceeding 25 characters in size
    tokens = [token for token in tokens if len(token)<=25]
    cleaned_docs_title.append(tokens)
    if idx % round(len(corpus_title)/10) == 0:
        print(idx)

#%%
print("Building the w2v")
# create empty word vectors for the words in vocabulary 
my_q = 300 # to match dim of GNews word vectors
mcount = 5
w2v_abstract = Word2Vec(size=my_q, min_count=mcount)
w2v_title = Word2Vec(size=my_q, min_count=mcount)

### fill gap ### # hint: use the build_vocab method
w2v_abstract.build_vocab(cleaned_docs_abstract)
w2v_title.build_vocab(cleaned_docs_title)

# load vectors corresponding to our vocabulary
w2v_abstract.intersect_word2vec_format(path_to_google_news + 'GoogleNews-vectors-negative300.bin.gz', binary=True)
w2v_title.intersect_word2vec_format(path_to_google_news + 'GoogleNews-vectors-negative300.bin.gz', binary=True)


#%% Feature Engineering
print("Feature Engineering test")
# test
# we need to compute the features for the testing set

overlap_title_test = []
temp_diff_test = []
comm_auth_test = []

comm_auth_prop_test = []
overlap_journal_test = []
WMD_abstract_test = []
WMD_title_test = []
   
counter = 0
for i in xrange(len(testing_set)):
    source = testing_set[i][0]
    target = testing_set[i][1]
    
    index_source = IDs.index(source)
    index_target = IDs.index(target)
        
    source_info = [element for element in node_info if element[0]==source][0]
    target_info = [element for element in node_info if element[0]==target][0]
    
    source_title = source_info[2].lower().split(" ")
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]
    
    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]
    
    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")
    
    overlap_title_test.append(len(set(source_title).intersection(set(target_title))))
    temp_diff_test.append(int(source_info[1]) - int(target_info[1]))
    comm_auth_test.append(len(set(source_auth).intersection(set(target_auth))))
    
    comm_auth_prop_test.append(len(set(source_auth).intersection(set(target_auth)))/(max(1.0,(len(source_auth)+len(target_auth))/2.0)))

    target_abstract = [elt for elt in target_info[5].split(' ') if elt not in stpwds]
    source_abstract = [elt for elt in source_info[5].split(' ') if elt not in stpwds]
    
    WMD_abstract_test.append(w2v_abstract.wmdistance(target_abstract,source_abstract))
    
    
    target_title = [elt for elt in target_info[2].split(' ') if elt not in stpwds]
    source_title = [elt for elt in source_info[2].split(' ') if elt not in stpwds]
    
    WMD_title_test.append(w2v_title.wmdistance(target_title,source_title))
    
    
    source_journal = source_info[4]
    target_journal = target_info[4]
    
    if source_journal != '' and target_journal != '':
        source_journal = source_journal.lower().split(" ")	
        source_journal = [token for token in source_journal if token not in stpwds]
        source_journal = [stemmer.stem(token) for token in source_journal]
        
        target_journal = target_journal.lower().split(" ")	
        target_journal = [token for token in target_journal if token not in stpwds]
        target_journal = [stemmer.stem(token) for token in target_journal]       
        
        overlap_journal_test.append(len(set(source_journal).intersection(set(target_journal))))
        
        target_journal = [elt for elt in target_info[4].split(' ') if elt not in stpwds]
        source_journal = [elt for elt in source_info[4].split(' ') if elt not in stpwds]
        
    else :
        overlap_journal_test.append(0.0)
    
   
    counter += 1
    if counter % 1000 == True:
        print(counter, "testing examples processsed")
        
# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
testing_features = np.array([overlap_title_test,temp_diff_test,comm_auth_test,comm_auth_prop_test,overlap_journal_test, WMD_abstract_test, WMD_title_test]).T

testing_features[np.isinf(testing_features)] = 0.0

# scale
testing_features = preprocessing.scale(testing_features)

testset = pd.DataFrame(data = testing_features, columns=["overlap_title", "temp_diff", "comm_auth", "comm_auth_prop", "overlap_journal", "WMD_abstract", "WMD_title"])
testset.to_csv('data/testset_full.csv')


#frac = 1.0
#print("Feature Engineering train with ",frac," of the train set")
# in this baseline we will train the model on only  frac % of the training set


# randomly select 5% of training set
#to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*frac)))
training_set_reduced = training_set #[training_set[i] for i in to_keep]

## basic features:
# number of overlapping words in title
overlap_title = []
# temporal distance between the papers
temp_diff = []
# number of common authors
comm_auth = []

## others features:
# proportion of common authors in the average total number of author
comm_auth_prop = []
# number of overlapping words in journal
overlap_journal = []
# WMD for the titles
WMD_title = []
# WMD for the abstracts
WMD_abstract = []


# Other ideas: cosine similarity, centroids similarity


counter = 0
for i in xrange(len(training_set_reduced)):
    source = training_set_reduced[i][0]
    target = training_set_reduced[i][1]
    
    index_source = IDs.index(source)
    index_target = IDs.index(target)
    
    source_info = [element for element in node_info if element[0]==source][0]
    target_info = [element for element in node_info if element[0]==target][0]
    
	# convert to lowercase and tokenize
    source_title = source_info[2].lower().split(" ")
	# remove stopwords
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]
    
    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]
    
    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")
    
    overlap_title.append(len(set(source_title).intersection(set(target_title))))
    temp_diff.append(int(source_info[1]) - int(target_info[1]))
    comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
    
    comm_auth_prop.append(len(set(source_auth).intersection(set(target_auth)))/(max(1.0,(len(source_auth)+len(target_auth))/2.0)))
    
    target_abstract = [elt for elt in target_info[5].split(' ') if elt not in stpwds]
    source_abstract = [elt for elt in source_info[5].split(' ') if elt not in stpwds]
    
    WMD_abstract.append(w2v_abstract.wmdistance(target_abstract,source_abstract))
    
    
    
    target_title = [elt for elt in target_info[2].split(' ') if elt not in stpwds]
    source_title = [elt for elt in source_info[2].split(' ') if elt not in stpwds]
    
    WMD_title.append(w2v_title.wmdistance(target_title,source_title))
    
    source_journal = source_info[4]
    target_journal = target_info[4]
    
    if source_journal != '' and target_journal != '':
        source_journal = source_journal.lower().split(" ")	
        source_journal = [token for token in source_journal if token not in stpwds]
        source_journal = [stemmer.stem(token) for token in source_journal]
        
        target_journal = target_journal.lower().split(" ")	
        target_journal = [token for token in target_journal if token not in stpwds]
        target_journal = [stemmer.stem(token) for token in target_journal]       
        
        overlap_journal.append(len(set(source_journal).intersection(set(target_journal))))
        
        target_journal = [elt for elt in target_info[4].split(' ') if elt not in stpwds]
        source_journal = [elt for elt in source_info[4].split(' ') if elt not in stpwds]
        
    else :
        #NB: It could be possible to use another encoding nan
        overlap_journal.append(0.0)
    
    counter += 1
    if counter % 1000 == True:
        print(counter, "training examples processsed")
        print "WMD_abstract :", WMD_abstract[-1]
        print "WMD_title :", WMD_title[-1]
        print "overlap_journal :", overlap_journal[-1]
        print "comm_auth_prop :", comm_auth_prop[-1]
        print "overlap_title :", overlap_title[-1]
        print "temp_diff :", temp_diff[-1]
        print "comm_auth :", comm_auth_prop[-1]

# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
training_features = np.array([overlap_title, temp_diff, comm_auth, comm_auth_prop, overlap_journal, WMD_abstract, WMD_title]).T

# NB: WMD could return inf, we treat this case to 0.0
training_features[np.isinf(training_features)] = 0.0

# scale
#training_features = np.nan_to_num(training_features)
training_features = preprocessing.scale(training_features)

# convert labels into integers then into column array
labels = [int(element[2]) for element in training_set_reduced]
labels = list(labels)
labels_array = np.array(labels)

trainset = pd.DataFrame(data = training_features, columns=["overlap_title", "temp_diff", "comm_auth", "comm_auth_prop", "overlap_journal", "WMD_abstract", "WMD_title"])
trainset['label'] = labels_array
trainset.to_csv('data/trainset_full.csv')
