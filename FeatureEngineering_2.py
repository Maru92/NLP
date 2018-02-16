import random
import numpy as np
import pandas as pd
import igraph
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
import nltk
import csv

from gensim.models.word2vec import Word2Vec
path_to_google_news = "../"
import re
import string

import os
print("old working dir",os.getcwd())
os.chdir('c:\\Users\\Marc\\desktop\\NLP\\NLP')
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

#%% data loading and preprocessing

# the columns of the data frame below are:
# (1) paper unique ID (integer)
# (2) publication year (integer)
# (3) paper title (string)
# (4) authors (strings separated by ,)
# (5) name of journal (optional) (string)
# (6) abstract (string) - lowercased, free of punctuation except intra-word dashes


testing_set = pd.read_csv('../data/test_both_j.csv')

with open("node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info  = list(reader)

IDs = [element[0] for element in node_info]

#%% TF-IDF
# compute TFIDF vector of each paper
corpus = [element[5] for element in node_info]

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

corpus_journal = [element[4] for element in node_info if element[4]!='']
cleaned_docs_journal = []
for idx, doc in enumerate(corpus_journal):
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
    cleaned_docs_journal.append(tokens)
    if idx % round(len(corpus_journal)/10) == 0:
        print(idx)


#%% TF-IDF Journal
tfidf_vect = TfidfVectorizer(min_df=1,
                             stop_words=None,
                             lowercase=False,
                             preprocessor=None)

tfidf_vect.fit([' '.join(elt) for elt in cleaned_docs_journal])


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

del cleaned_docs_title, cleaned_docs_abstract

# load vectors corresponding to our vocabulary
w2v_abstract.intersect_word2vec_format(path_to_google_news + 'GoogleNews-vectors-negative300.bin.gz', binary=True)
w2v_title.intersect_word2vec_format(path_to_google_news + 'GoogleNews-vectors-negative300.bin.gz', binary=True)


#%% Feature Engineering
print("Feature Engineering test")

comm_auth_prop_test = []
overlap_journal_test = []
WMD_abstract_test = []
WMD_title_test = []
comm_title_prop_test = []
#cosine_journal_test = []

counter = 0
for i in xrange(testing_set.shape[0]):

    source_title = testing_set['Title_source'][i].lower().split(" ")
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]

    target_title = testing_set['Title_target'][i].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]

    inter_2 = len(set(source_title).intersection(set(target_title)))
    comm_title_prop_test.append(inter_2/(0.0+len(source_title)+len(target_title)-inter_2))


    source_auth = testing_set['Authors_source'][i]
    target_auth = testing_set['Authors_target'][i]
    if type(target_auth) == str and type(source_auth) == str:
        target_auth = target_auth.split(",")
        source_auth = source_auth.split(",")
        inter = len(set(source_auth).intersection(set(target_auth)))
        comm_auth_prop_test.append(inter/(0.0+len(source_auth)+len(target_auth)-inter))
    else:
        comm_auth_prop_test.append(0.0)

    target_abstract = [elt for elt in testing_set['Abstract_target'][i].split(' ') if elt not in stpwds]
    source_abstract = [elt for elt in testing_set['Abstract_source'][i].split(' ') if elt not in stpwds]

    WMD_abstract_test.append(w2v_abstract.wmdistance(target_abstract,source_abstract))


    target_title = [elt for elt in testing_set['Title_target'][i].split(' ') if elt not in stpwds]
    source_title = [elt for elt in testing_set['Title_source'][i].split(' ') if elt not in stpwds]

    WMD_title_test.append(w2v_title.wmdistance(target_title,source_title))


    source_journal = testing_set['Journal_source'][i]
    target_journal = testing_set['Journal_target'][i]

    if type(target_journal) == str and type(source_journal) == str:
        source_journal = source_journal.lower().split(" ")
        source_journal = [token for token in source_journal if token not in stpwds]
        source_journal = [stemmer.stem(token) for token in source_journal]

        target_journal = target_journal.lower().split(" ")
        target_journal = [token for token in target_journal if token not in stpwds]
        target_journal = [stemmer.stem(token) for token in target_journal]

        overlap_journal_test.append(len(set(source_journal).intersection(set(target_journal))))

        #vect_source = tfidf_vect.transform(source_journal)
        #vect_target = tfidf_vect.transform(target_journal)
        #cosine_journal_test.append(cosine_similarity(vect_source,vect_target))

    else :
        overlap_journal_test.append(0.0)
        #cosine_journal_test.append(0.0)


    counter += 1
    if counter % 1000 == True:
        print(counter, "testing examples processsed")
        print("WMD_abstract :", WMD_abstract_test[-1])
        print("WMD_title :", WMD_title_test[-1])
        print("overlap_journal :", overlap_journal_test[-1])
        print("comm_title_prop :", comm_title_prop_test[-1])
        print("comm_auth_prop :", comm_auth_prop_test[-1])
        #print("cosine_journal :", cosine_journal_test[-1])

# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
testing_features = np.array([comm_auth_prop_test, overlap_journal_test, WMD_abstract_test, WMD_title_test, comm_title_prop_test]).T #, cosine_journal_test]).T

del comm_auth_prop_test, overlap_journal_test, WMD_abstract_test, WMD_title_test, comm_title_prop_test#, cosine_journal_test

testing_features[np.isinf(testing_features)] = 0.0

# scale
testing_features = preprocessing.scale(testing_features)

testing_set['Common_authors_prop'] = testing_features[:,0]
testing_set['Overlap_journal'] = testing_features[:,1]
testing_set['WMD_abstract'] = testing_features[:,2]
testing_set['WMD_title'] = testing_features[:,3]
testing_set['Common_title_prop'] = testing_features[:,4]
#testing_features['Tfidf_cosine_journal'] = testing_features[:,5]

print("Writing ...")
testing_set.to_csv('../data/test_fusion_15_02.csv')

del testing_set, testing_features

#%%
print("Import Training: ")
training_set = pd.read_csv('../data/train_both.csv')

comm_auth_prop = []
overlap_journal = []
WMD_title = []
WMD_abstract = []
comm_title_prop = []
#cosine_journal = []

counter = 0
for i in xrange(training_set.shape[0]):

    source_title = training_set['Title_source'][i].lower().split(" ")
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]

    target_title = training_set['Title_target'][i].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]

    inter_2 = len(set(source_title).intersection(set(target_title)))
    comm_title_prop.append(inter/(0.0+len(source_title)+len(target_title)-inter_2))

    source_auth = training_set['Authors_source'][i]
    target_auth = training_set['Authors_target'][i]
    if type(target_auth) == str and type(source_auth) == str:
        target_auth = target_auth.split(",")
        source_auth = source_auth.split(",")
        inter = len(set(source_auth).intersection(set(target_auth)))
        comm_auth_prop.append(inter/(0.0+len(source_auth)+len(target_auth)-inter))
    else:
        comm_auth_prop.append(0.0)

    target_abstract = [elt for elt in training_set['Abstract_target'][i].split(' ') if elt not in stpwds]
    source_abstract = [elt for elt in training_set['Abstract_source'][i].split(' ') if elt not in stpwds]

    WMD_abstract.append(w2v_abstract.wmdistance(target_abstract,source_abstract))


    target_title = [elt for elt in training_set['Title_target'][i].split(' ') if elt not in stpwds]
    source_title = [elt for elt in training_set['Title_source'][i].split(' ') if elt not in stpwds]

    WMD_title.append(w2v_title.wmdistance(target_title,source_title))

    source_journal = training_set['Journal_source'][i]
    target_journal = training_set['Journal_target'][i]

    if type(target_journal) == str and type(source_journal) == str:
        source_journal = source_journal.lower().split(" ")
        source_journal = [token for token in source_journal if token not in stpwds]
        source_journal = [stemmer.stem(token) for token in source_journal]

        target_journal = target_journal.lower().split(" ")
        target_journal = [token for token in target_journal if token not in stpwds]
        target_journal = [stemmer.stem(token) for token in target_journal]

        overlap_journal.append(len(set(source_journal).intersection(set(target_journal))))

        #vect_source = tfidf_vect.transform(source_journal)
        #vect_target = tfidf_vect.transform(target_journal)
        #cosine_journal.append(cosine_similarity(vect_source,vect_target))

    else :
        overlap_journal.append(0.0)
        #cosine_journal.append(0.0)

    counter += 1
    if counter % 1000 == True:
        print(counter, "training examples processsed")
        print("WMD_abstract :", WMD_abstract[-1])
        print("WMD_title :", WMD_title[-1])
        print("overlap_journal :", overlap_journal[-1])
        print("comm_title_prop :", comm_title_prop[-1])
        print("comm_auth_prop :", comm_auth_prop[-1])
        #print("cosine_journal :", cosine_journal[-1])

# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
training_features = np.array([comm_auth_prop, overlap_journal, WMD_abstract, WMD_title,comm_title_prop]).T#,cosine_journal]).T

del comm_auth_prop, overlap_journal, WMD_abstract, WMD_title, comm_title_prop #,cosine_journal

# NB: WMD could return inf, we treat this case to 0.0
training_features[np.isinf(training_features)] = 0.0

# scale
#training_features = np.nan_to_num(training_features)
training_features = preprocessing.scale(training_features)

training_set['Common_authors_prop'] = training_features[:,0]
training_set['Overlap_journal'] = training_features[:,1]
training_set['WMD_abstract'] = training_features[:,2]
training_set['WMD_title'] = training_features[:,3]
training_set['Common_title_prop'] = training_features[:,4]
#training_set['Tfidf_cosine_journal'] = training_features[:,5]

print("Writing ...")
training_set.to_csv('../data/train_fusion_15_02.csv')
