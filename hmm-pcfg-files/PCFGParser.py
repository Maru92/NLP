import optparse
import numpy as np

#%%
def get_sentence(sent_file):
    with open(sent_file, 'r') as my_file: 
        test_sents = my_file.read().splitlines()
    
    test_sents_2 = []
    for line in test_sents:
        test_sents_2.append(line.split(' '))
    
    return test_sents_2

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
    known_vocab = set([])
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
                known_vocab.add(table[1])
        elif len(table) == 4:
            rule = tuple([table[0],table[1]+" "+table[2]])
            grammar.append(rule)
            rule_probs[rule] = float(table[3])
        else:
            print("Load error : ", table)
    return set(grammar), rule_probs, known_vocab

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
            #bin_set.add((rules[0], c, b))
    return list(bin_set)

# We delete the unary rule part, we don't have it
# We change to have prob in log space
def cky(words,grammar,rule_probabilities,knwon_vocab):
    '''
    cky() takes a sentence and parses it according to the provided grammar.
    :param words: words in the sentence (list)
    :param grammar: list of grammar rules
    :param rule_probabilities: the probabilities of a given grammar rule (dictionary)
    :return: GrammarTree: parse tree with highest probability
    '''

    non_terms = list(get_non_terms(grammar))
    score = np.full((len(words)+1,len(words)+1,len(non_terms)),-np.inf) 
    back = [[[-1 for i in range(len(non_terms))] for j in range(len(words)+1)] for k in range(len(words)+1)]
    rule_index = {}

    for i,word in enumerate(words):
        if word not in knwon_vocab:
            rule = tuple(['NONE',word])
            grammar.add(rule)
            rule_probs[rule] = 0.0

        for j,A in enumerate(non_terms):
            r = A, word   # Here change due to encode of word
            if r in grammar:
                score[i,i+1,j] = rule_probabilities[r]
                rule_index[A] = j
            else:
                rule_index[A] = j

    binary_rules = get_binary_rules(grammar)
    for span in range(2,len(words)+1):
        for begin in range(len(words)+1-span):
            end = begin + span
            for split in range(begin+1, end):
                for rule in binary_rules:
                    if begin == 0 and end == len(words) and rule[0]!='S':
                        continue
                    a, b, c = rule_index[rule[0]], rule_index[rule[1]], rule_index[rule[2]]
                    concat_rule = rule[0], ' '.join((rule[1], rule[2]))
                    #if concat_rule in grammar:
                    prob = score[begin,split,b] + score[split,end,c] + rule_probabilities[concat_rule]
                    if prob > score[begin,end,a]:
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
    root_index = np.argmax(score[0,len(score)-1])
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
        
def get_string_tree(head, sentence, count = 0):
    result = ""
    temp_left = ""
    temp_right = ""
    if head.left == None and head.right == None:
        result = "( "+head.data+" "+sentence[count]+" )"
        count +=1
    else:
        if head.left != None:
            temp_left, count = get_string_tree(head.left,sentence,count)
        if head.right != None:
            temp_right, count = get_string_tree(head.right,sentence,count)
        result = "( "+head.data+" "+temp_left+" "+temp_right+" )"
    return result, count

#%%
if __name__ == '__main__':
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--pcfg", dest="pcfg", default="pcfg")
    opt_parser.add_option("--data", dest="data", default="dev_sents")
    opt_parser.add_option("--output", dest="output", default="candidate-parses")

    (options, args) = opt_parser.parse_args()
    options = vars(options)

    print("PCFGParser options:")
    for opt in options:
        print("  %-12s: %s", (opt, options[opt]))
    print("")
    
    grammar, rule_probs, known_vocab = get_grammar_prob(options['pcfg'])
    sentences = get_sentence(options['data'])
    
    results = []
    for sentence in sentences:
        print("New sentence ...")
        parse_tree = cky(sentence,grammar,rule_probs,known_vocab)
        res, count = get_string_tree(parse_tree,sentence)
        if count != len(sentence):
            print("Error of labeling")
        print("Parsing finished ...")
        results.append(res)         
    
    with open(options['output'], 'w') as my_file:
        for line in results:
            my_file.write(line+'\n')
    