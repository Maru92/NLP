import os
os.chdir("C:\Users\Marc\Desktop\NLP\Assignment\hmm-pcfg-files")

import numpy as np

#%%
with open('pcfg', 'r') as my_file: 
    pcfg = my_file.read().splitlines()

with open('test_sents', 'r') as my_file: 
    test_sents = my_file.read().splitlines()
    
with open('dev_sents', 'r') as my_file: 
    dev_sents = my_file.read().splitlines()
    
#%%
test_sents_2 = []
count = 0
for line in test_sents:
    test_sents_2.append(line.split(' '))
    
dev_sents_2 = []
count = 0
for line in dev_sents:
    dev_sents_2.append(line.split(' '))

#%%
def get_grammar_prob(filename):
    '''
    get_grammar() reads in a text file of CNF grammar rules.
    :param filename: text file in CNF format i.e.
                            S -> VP
                            S -> NP VP
    :return: python list with grammar rules as tuples
    '''
    with open(filename,'r') as my_file:
        pcfg = my_file.read().splitlines()
        
    rule_probs = {}
    grammar = []
    known_vocab = []
    for line in pcfg:
        tables = line.split('\t')
        table = []
        for t in tables:
            table = table + t.split(" ")
        if len(table) == 3:
            rule = tuple([table[0],table[1]])
            grammar.append(rule)
            rule_probs[rule] = float(table[2])
            if table[1] not in known_vocab:
                known_vocab.append(table[1])
        elif len(table) == 4:
            rule = tuple([table[0],table[1]+" "+table[2]])
            grammar.append(rule)
            rule_probs[rule] = float(table[3])
        else:
            print "Load error : ", table 
    return grammar, rule_probs, known_vocab

def get_binary_rules(grammar):
    '''
    get_binary_rules() searches through the grammar binary rules and returns them in a list
    :param grammar: dictionary of grammar rules
    :return: list of binary grammar rules
    '''
    bin_set = set()
    for rules in grammar:
        if len(rules[1].split(' ')) == 2:
            b, c = rules[1].split(' ')
            bin_set.add((rules[0],b,c))
            bin_set.add((rules[0], c, b))
    return list(bin_set)

# We delete the unary rule part, we don't have it
# We change to have prob in log space
def cky(words,grammar,rule_probabilities,knwon_vocab,states):
    '''
    cky() takes a sentence and parses it according to the provided grammar.
    :param words: words in the sentence (list)
    :param grammar: list of grammar rules
    :param rule_probabilities: the probabilities of a given grammar rule (dictionary)
    :return: GrammarTree: parse tree with highest probability
    '''

    non_terms = list(get_non_terms(grammar))
    score = np.full((len(words)+1,len(words)+1,len(non_terms)),-np.inf)
    back = [[[-1 for i in xrange(len(states))] for j in xrange(len(words)+1)] for k in xrange(len(words)+1)]

    for i,word in enumerate(words):
        if word not in knwon_vocab:
            print "Word unknown : ", word, " . Adding the rule."
            rule = tuple(['NONE',word])
            grammar.append(rule)
            rule_probs[rule] = 0.0
        print word, " : Word number ", i," over ", len(words)

        for A in non_terms:
            r = A, word   # Here change due to encode of word
            if r in grammar:
                score[i,i+1,states[A]] = rule_probabilities[r]

    binary_rules = get_binary_rules(grammar)
    for span in xrange(2,len(words)+1):
        print "New span : ", span
        for begin in xrange(len(words)+1-span):
            end = begin + span
            for split in xrange(begin+1, end):
                for rule in binary_rules:
                    if begin == 0 and end == len(words) and rule[0]!='S':
                        continue
                    a, b, c = states[rule[0]], states[rule[1]], states[rule[2]]
                    concat_rule = rule[0], ' '.join((rule[1], rule[2]))
                    if concat_rule in grammar:
                        prob = score[begin,split,b] + score[split,end,c] + rule_probabilities[concat_rule]
                    else:
                        continue
                    if prob > score[begin][end][a]:
                        print "New max for prob: ", prob
                        print "Begin ",begin," , End ",end," , Rule index ",a
                        score[begin,end,a] = prob
                        back[begin][end][a] = split, b, c

    return get_parse_tree(score,back,non_terms)

def get_parse_tree(score, back, non_terms):
    '''
    get_parse_tree() calls the build_tree() method
    :param score: score matrix
    :param back: backpointer matrix
    :param non_terms: list of non_terminals
    :return: GrammarTree the final parse tree
    '''
    root_index = score[0][len(score)-1].index(max(score[0][len(score)-1]))
    tree = build_tree(0,len(score)-1,root_index,back,non_terms)
    return tree

def build_tree(start,end,idx,back,non_terms):
    '''
    build_tree() builds tree from the backpointer matrix obtained in the cky() function
    :param start: start index for tree
    :param end: end index for tree
    :param idx: index used to find non_terminal
    :param back: the backpointer matrix
    :param non_terms: a list of non-terminals
    :return:
    '''
    tree = GrammarTree(non_terms[idx])
    node = back[start][end][idx]
    if isinstance(node,tuple):
        split,left_rule,right_rule = node
        tree.insertLeft(build_tree(start,split,left_rule,back,non_terms))
        tree.insertRight(build_tree(split,end,right_rule,back,non_terms))
        return tree
    else:
        if node>0:
            tree.insertLeft(GrammarTree(non_terms[node]))
        return tree
    
class GrammarTree(object):
    '''
    Tree data structure used to represent the grammar tree output generated by the cky algorithm
    '''
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

    def insertLeft(self, new_node):
            self.left = new_node

    def insertRight(self, new_node):
            self.right = new_node
            
def init_parse_triangle(number_of_words,number_of_nonterm, fill_value=0):
    '''
    :param number_of_words:
    :param number_of_nonterm:
    :param fill_value: the value used to fill the parse triangle
    :return: a parse triangle with the params used as dimensions
    '''
    return [[[fill_value for i in xrange(number_of_nonterm)] for j in xrange(number_of_words)] for k in xrange(number_of_words)]

def get_non_terms(grammar):
    '''
    get_non_terms() returns a set of the non-terminal rules
    :param grammar: dictionary of grammar rules
    :return: a set of non_terms from the grammar
    '''
    non_terms = set()

    for rules in set(grammar):
        non_terms.add(rules[0])

    return non_terms

from collections import deque

def print_level_order(head, queue = deque()):
    '''
    Helper method used to print tree to console.
    :return: n/a
    '''
    if isinstance(head,str):
        print head
        return
    print head.data
    [queue.append(node) for node in [head.left, head.right] if node]
    if queue:
        print_level_order(queue.popleft(), queue)
        
def get_string_tree(head, sentence, count = 0):
    result = ""
    temp_left = ""
    temp_right = ""
    if head.left == None and head.right == None:
        result = "( "+head.data+" "+sentence[count]+" )"
#        print "Leaf node : ", result
        count +=1
    else:
        if head.left != None:
            temp_left, count = get_string_tree(head.left,sentence,count)
        if head.right != None:
            temp_right, count = get_string_tree(head.right,sentence,count)
#        print head.data
#        print temp_left
#        print temp_right
        result = "( "+head.data+" "+temp_left+" "+temp_right+" )"
    return result, count
#%%
grammar_bis, rule_probs, known_vocab = get_grammar_prob('pcfg')

#%%
non_terms = list(get_non_terms(grammar_bis))

#%%
def complete_states_dict(states,grammar):
    inner_states = states
    start = len(states)
    non_terms = list(get_non_terms(grammar))
    count = len(states)
    already_seen = 0
    for state in non_terms:
        if state not in inner_states:
            inner_states[state] = count
            count += 1
        else:
            already_seen += 1
    print "States already seen", already_seen
    print "States in dict de base", start   
    return inner_states

#%%
x = {}
states_bis = complete_states_dict(x,grammar_bis)

#%%
sentence = dev_sents_2[2]
parse_tree = cky(sentence,grammar_bis,rule_probs,known_vocab,states_bis)
res, count = get_string_tree(parse_tree,sentence)
print res

#%%
res, count = get_string_tree(parse_tree,sentence)

#%%
parses_pred_dev = []
for sent in dev_sents_2:
    print "New sentence ..."
    parse_tree = cky(sentence,grammar_bis,rule_probs,known_vocab)
    res, count = get_string_tree(parse_tree,sentence)
    print "Parsing finished ..."
    parses_pred_dev.append(res)

with open('candidate-parses-dev', 'w') as my_file:
    for line in parses_pred_dev:
        my_file.write(line+'\n')

#%%
parses_pred_test = []
for sent in test_sents_2:
    parse_tree = cky(sentence,grammar_bis,rule_probs,known_vocab)
    res, count = get_string_tree(parse_tree,sentence)
    if count != len(sentence):
        print "Error of labeling"
    parses_pred_test.append(res)

with open('candidate-parses', 'w') as my_file:
    for line in parses_pred_test:
        my_file.write(line+'\n')