import numpy as np
import optparse
import POSTAG as postag
import PCFGParser as pcfg

#%%
# Return the list of the postag (with their encoding) from the parser tree
def get_postag_from_parser(head,states):
    result = []
    temp_left = []
    temp_right = []
    if head.left == None and head.right == None:
        result = [states[head.data]]
    else:
        if head.left != None:
            temp_left = get_postag_from_parser(head.left,states)
        if head.right != None:
            temp_right = get_postag_from_parser(head.right,states)
        for l in temp_left:
            result.append(l)
        for r in temp_right:
            result.append(r)
    return result

def update_constraints(u,k,y_pcfg,z_hmm):
    # http://www.cs.columbia.edu/~mcollins/acltutorial.pdf
    eta = 0.001/(1+k)
    ind = (y_pcfg != z_hmm)
    u[ind] -= eta*(y_pcfg[ind] - z_hmm[ind])
    return u

def get_string_postag(postag_hmm,inv_states):    
    pred = ""
    start = True
    for etat in postag_hmm:
        if start:
            pred = pred + inv_states[etat]
            start = False
        else:
            pred = pred + " " + inv_states[etat]
        
    return pred

def complete_states_dict(states,grammar):
    inner_states = states.copy()
    non_terms = list(pcfg.get_non_terms(grammar))
    count = len(states)
    already_seen = 0
    for state in non_terms:
        if state not in inner_states:
            inner_states[state] = count
            count += 1
        else:
            already_seen += 1
    return inner_states

def cky_dual(words,grammar,rule_probabilities,knwon_vocab,states,u):
    non_terms = list(pcfg.get_non_terms(grammar))
    score = np.full((len(words)+1,len(words)+1,len(non_terms)),-np.inf) 
    back = [[[-1 for i in range(len(states))] for j in range(len(words)+1)] for k in range(len(words)+1)]

    for i,word in enumerate(words):
        if word not in knwon_vocab:
            print("Word unknown : ", word, " . Adding the rule.")
            rule = tuple(['NONE',word])
            grammar.add(rule)
            rule_probs[rule] = 0.0

        for A in non_terms:
            r = A, word   # Here change due to encode of word
            if r in grammar:
                score[i,i+1,states[A]] = rule_probabilities[r] - u[i][states[A]]

    binary_rules = pcfg.get_binary_rules(grammar)
    for span in range(2,len(words)+1):
        print("New span : ", span)
        for begin in range(len(words)+1-span):
            end = begin + span
            for split in range(begin+1, end):
                for rule in binary_rules:
                    if begin == 0 and end == len(words) and rule[0]!='S':
                        continue
                    a, b, c = states[rule[0]], states[rule[1]], states[rule[2]]
                    concat_rule = rule[0], ' '.join((rule[1], rule[2]))
                    #if concat_rule in grammar:
                    prob = score[begin,split,b] + score[split,end,c] + rule_probabilities[concat_rule]
                    if prob > score[begin,end,a]:
                        score[begin,end,a] = prob
                        back[begin][end][a] = split, b, c

    return pcfg.get_parse_tree(score,back,non_terms)

# The lagrangian is juste an additive term to take into account
# It penalizes the emissions probabilities
def scores_sent_dual(sent, initial_probs, transition_probs, final_probs, emission_probs, observations, states, u):
    S = len(initial_probs)
    emission_scores = np.zeros([len(sent), S]) + postag.logzero()
    transition_scores = np.zeros([len(sent)-1, S, S]) + postag.logzero()
    
    initial_scores = initial_probs
    final_scores = final_probs
    
    for pos in range(len(sent)):
        word = sent[pos].lower()
        if word in observations.keys() :
            # We keep -inf for the indexes where we have no information
            ind = (emission_probs[observations[word], :] != 0)
            emission_scores[pos, ind] = emission_probs[observations[word], ind] + u[pos, ind]
        else:
            emission_scores[pos, states["NONE"]] = 0 + u[pos, states["NONE"]]
            print(word, " not in dictionnary. Uncoded NONE !")
        if pos > 0:
            transition_scores[pos-1, :, :] = transition_probs
        
    return initial_scores, transition_scores, final_scores, emission_scores
    
#%%
if __name__ == '__main__':
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--pcfg", dest="pcfg", default="pcfg")
    opt_parser.add_option("--data", dest="data", default="dev_sents")
    opt_parser.add_option("--output_parses", dest="output_parses", default="candidate-parses-dual")
    opt_parser.add_option("--transition", dest="transition", default="hmm_trans")
    opt_parser.add_option("--emission", dest="emission", default = "hmm_emits")
    opt_parser.add_option("--output_postags", dest="output_postags", default="candidate-postags-dual")
    opt_parser.add_option("--viterbi", dest="viterbi", default="Y")
    opt_parser.add_option("--iteration", dest="iteration", default="1")

    (options, args) = opt_parser.parse_args()
    options = vars(options)

    print("Dual options:")
    for opt in options:
        print("  %-12s: %s", (opt, options[opt]))
    print("")
    
    K = int(options['iteration'])
    
    # Postag initialization
    states, observations, initial_probs, transition_probs, final_probs, emission_probs = postag.get_transition_emission(options['transition'],options['emission'])
    # Python 2
    #inv_states = {v: k for k, v in states.iteritems()}
    # Python 3
    inv_states = {v: k for k, v in states.items()}
    viterbi = (options['viterbi'] == "Y")
    
    # PCFG initialization
    grammar, rule_probs, known_vocab = pcfg.get_grammar_prob(options['pcfg'])
    inner_states = complete_states_dict(states,grammar)    
    
    # Sentences
    sentences = pcfg.get_sentence(options['data'])
    
    best_hmm = []
    best_pcfg = []
    for sent in sentences:
        # Initialization
        u = np.zeros((len(sent),len(states)))
        
        for k in range(K):
            print("New iteration : ",k)             
            # Initialization
            y_pcfg = np.zeros((len(sent),len(states)))
            z_hmm = np.zeros((len(sent),len(states)))
            
            
            if viterbi:
                initial_scores, transition_scores, final_scores, emission_scores = scores_sent_dual(sent, initial_probs, transition_probs, final_probs, emission_probs, observations, states, u)
                postag_hmm, _ = postag.run_viterbi(initial_scores, transition_scores, final_scores, emission_scores)
                
            else:
                initial_scores, transition_scores, final_scores, emission_scores = scores_sent_dual(sent, initial_probs, transition_probs, final_probs, emission_probs, observations, states, u)
                state_posteriors, _, _ = postag.compute_posteriors(initial_scores,
                                                                 transition_scores,
                                                                 final_scores,
                                                                 emission_scores)
                postag_hmm = np.argmax(state_posteriors, axis=1)
            
            for i,x in enumerate(postag_hmm):
                z_hmm[i,x] = 1
            
            parse_tree = cky_dual(sent,grammar,rule_probs,known_vocab,inner_states,u)
            postag_pcfg = get_postag_from_parser(parse_tree,inner_states)
            
            for i,x in enumerate(postag_pcfg):
                if x < y_pcfg.shape[1]:
                    y_pcfg[i,x] = 1
                else:
                    print("Postag PCFG is inner not leaf")
            
            
            
            if np.sum(y_pcfg != z_hmm) == 0:
                print("Postag and PCFG agreed !")
                parsed_sent, _ = pcfg.get_string_tree(parse_tree,sent)
                best_pcfg.append(parsed_sent)
                best_hmm.append(get_string_postag(postag_hmm,inv_states))
                break
            else:
                u = update_constraints(u,k,y_pcfg,z_hmm)
            if k == K-1:
                print("Constraints not satisfied, end of iterations !")
                print("Number of differences : ", np.sum(y_pcfg != z_hmm))
                parsed_sent, _ = pcfg.get_string_tree(parse_tree,sent)
                best_pcfg.append(parsed_sent)
                best_hmm.append(get_string_postag(postag_hmm,inv_states))
       
    with open(options['output_parses'], 'w') as my_file:
        for line in best_pcfg:
            my_file.write(line+'\n')
    
    with open(options['output_postags'], 'w') as my_file:
        for line in best_hmm:
            my_file.write(line+'\n')
