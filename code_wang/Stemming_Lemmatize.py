import random
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
import nltk
import csv
from nltk.corpus import wordnet


from gensim.models.word2vec import Word2Vec
path_to_google_news = "../../wv/"
import re
import string

path = '../../data/'

punct = string.punctuation.replace('-','')
my_regex = re.compile(r"(\b[-]\b)|[\W_]")
#my_regex = re.compile(r"(\b\b)|[\W_]")

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None # for easy if-statement 
    
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
#nltk.download('punkt') # for tokenization
#nltk.download('stopwords')
#nltk.download('maxent_treebank_pos_tagger')
#stpwds = set(nltk.corpus.stopwords.words("english"))
# Other Approach of stop word to test
with open(path+'smart_stopwords.txt', 'r') as my_file:
    stpwds = my_file.read().splitlines()
PorterStemmer = nltk.stem.PorterStemmer()
SnowballStemmer = nltk.stem.SnowballStemmer("english")
WordNetLemmatizer = nltk.stem.wordnet.WordNetLemmatizer()


with open(path+"node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info  = list(reader)
    node_info = node_info[:10]

IDs = [element[0] for element in node_info]

corpus = [element[5] for element in node_info]
print("Cleaning the docs")
print("Starting abstract")

cleaned_docs_abstract = []
Porter_docs_abstract = []
Snowball_docs_abstract = []
WordNetLemmatize_docs_abstract = []
for idx, doc in enumerate(corpus):
    # clean
    doc = clean_string(doc, punct, my_regex, to_lower=True)
    # tokenize (split based on whitespace)
        #tokens = doc.split(" ")
    tokens = nltk.word_tokenize(doc)
    # remove stopwords
    tokens = [token for token in tokens if token not in stpwds]
    # remove digits
    tokens = [''.join([elt for elt in token if not elt.isdigit()]) for token in tokens]
    tokens = [token for token in tokens if len(token)>2]
    Portertokens = [PorterStemmer.stem(token) for token in tokens]
    PorterString = ' '.join(Portertokens)
    Snowballtokens = [SnowballStemmer.stem(token) for token in tokens]
    SnowballString = ' '.join(Snowballtokens)
    word_pos = nltk.pos_tag(tokens)
    Lemmatokens = [ WordNetLemmatizer.lemmatize(token,get_wordnet_pos(pos)) if get_wordnet_pos(pos) else  WordNetLemmatizer.lemmatize(token) for (token,pos) in word_pos]
    LemmaString = ' '.join(Lemmatokens)
    Porter_docs_abstract.append(PorterString)
    Snowball_docs_abstract.append(SnowballString)
    WordNetLemmatize_docs_abstract.append(LemmaString)
    '''
    # remove tokens shorter than 3 characters in size
    tokens = [token for token in tokens if len(token)>2]
    # remove tokens exceeding 25 characters in size
    tokens = [token for token in tokens if len(token)<=25]
    '''
    cleaned_docs_abstract.append(tokens)
    '''
    if idx % round(len(corpus)/10) == 0:
        print(idx)
    '''
print("Done Abstract")
print("Start title")

corpus_title = [element[2] for element in node_info]
cleaned_docs_title = []
Porter_docs_title = []
Snowball_docs_title = []
WordNetLemmatize_docs_title = []
for idx, doc in enumerate(corpus_title):
    # clean
    doc = clean_string(doc, punct, my_regex, to_lower=True)
    # tokenize (split based on whitespace)
        #tokens = doc.split(" ")
    tokens = nltk.word_tokenize(doc)
    # remove stopwords
    tokens = [token for token in tokens if token not in stpwds]
    # remove digits
    tokens = [''.join([elt for elt in token if not elt.isdigit()]) for token in tokens]
    tokens = [token for token in tokens if len(token)>2]
    Portertokens = [PorterStemmer.stem(token) for token in tokens]
    PorterString = ' '.join(Portertokens)
    Snowballtokens = [SnowballStemmer.stem(token) for token in tokens]
    SnowballString = ' '.join(Snowballtokens)
    word_pos = nltk.pos_tag(tokens)
    Lemmatokens = [ WordNetLemmatizer.lemmatize(token,get_wordnet_pos(pos)) if get_wordnet_pos(pos) else  WordNetLemmatizer.lemmatize(token) for (token,pos) in word_pos]
    LemmaString = ' '.join(Lemmatokens)
    Porter_docs_title.append(PorterString)
    Snowball_docs_title.append(SnowballString)
    WordNetLemmatize_docs_title.append(LemmaString)
    '''
    # remove tokens shorter than 3 characters in size
    tokens = [token for token in tokens if len(token)>2]
    # remove tokens exceeding 25 characters in size
    tokens = [token for token in tokens if len(token)<=25]
    '''
    cleaned_docs_title.append(tokens)
    '''
    if idx % round(len(corpus)/10) == 0:
        print(idx)
    '''

print("Done Title")
print("Start Journal")
corpus_journal = [element[4] for element in node_info]
cleaned_docs_journal = []
Porter_docs_journal = []
Snowball_docs_journal = []
WordNetLemmatize_docs_journal = []
for idx, doc in enumerate(corpus_journal): 
    # clean
    doc = clean_string(doc, punct, my_regex, to_lower=True)
    # tokenize (split based on whitespace)
    #tokens = doc.split(" ")
    tokens = nltk.word_tokenize(doc)
    # remove stopwords
    tokens = [token for token in tokens if token not in stpwds]
    # remove digits
    tokens = [''.join([elt for elt in token if not elt.isdigit()]) for token in tokens]
    tokens = [token for token in tokens if len(token)>2]
    Portertokens = [PorterStemmer.stem(token) for token in tokens]
    PorterString = ' '.join(Portertokens)
    Snowballtokens = [SnowballStemmer.stem(token) for token in tokens]
    SnowballString = ' '.join(Snowballtokens)
    word_pos = nltk.pos_tag(tokens)
    Lemmatokens = [ WordNetLemmatizer.lemmatize(token,get_wordnet_pos(pos)) if get_wordnet_pos(pos) else  WordNetLemmatizer.lemmatize(token) for (token,pos) in word_pos]
    LemmaString = ' '.join(Lemmatokens)
    Porter_docs_journal.append(PorterString)
    Snowball_docs_journal.append(SnowballString)
    WordNetLemmatize_docs_journal.append(LemmaString)
    
    '''
    # remove tokens shorter than 3 characters in size
    tokens = [token for token in tokens if len(token)>2]
    # remove tokens exceeding 25 characters in size
    tokens = [token for token in tokens if len(token)<=25]
    '''
    cleaned_docs_journal.append(tokens)
    '''
    if idx % round(len(corpus)/10) == 0:
        print(idx)
    '''
    
data = pd.DataFrame()
data['id'] = IDs
data['publication_year']=[elt[1] for elt in node_info]
data['title'] = corpus_title
data['author'] = [elt[3] for elt in node_info]
data['journal'] = corpus_journal
data['abstract'] = corpus
data['Porter_title'] = Porter_docs_title
data['Porter_journal'] = Porter_docs_journal
data['Porter_abstract'] = Porter_docs_abstract
data['Snowball_title'] = Snowball_docs_title
data['Snowball_journal'] = Snowball_docs_journal
data['Snowballr_abstract'] = Snowball_docs_abstract
data['Lemmatize_title'] = WordNetLemmatize_docs_title
data['Lemmatize_journal'] = WordNetLemmatize_docs_journal
data['Lemmatize_abstract'] = WordNetLemmatize_docs_abstract

data.to_csv(path+'node_info_test.csv', index=False)

