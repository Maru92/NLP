import optparse
import numpy as np

#%%
def get_transition_emission(trans_file,emis_file):
    #Transition
    with open(trans_file, 'r') as my_file: 
        hmm_trans = my_file.read().splitlines()
        
    hmm_trans_2 = []
    for line in hmm_trans:
        hmm_trans_2.append(line.split('\t'))
        if(len(hmm_trans_2[-1]) != 3):
            print("Error")
            
    states = {}
    count = 0
    for line in hmm_trans_2:
        state = line[0]
        if state not in states.keys() and state != "sentence_boundary":
            states[state] = count
            count += 1
        state = line[1]
        if state not in states.keys() and state != "sentence_boundary":
            states[state] = count
            count += 1
                
    #Emission
    with open(emis_file, 'r') as my_file: 
        hmm_emit = my_file.read().splitlines()
    
    hmm_emit_2 = []
    for line in hmm_emit:
        hmm_emit_2.append(line.split('\t'))
        if(len(hmm_emit_2[-1]) != 3):
            print("Error")
            
    observations = {}
    count = 0
    for line in hmm_emit_2:
        obs = line[1].lower()
        if obs not in observations.keys():
            observations[obs] = count
            count += 1
                
    #Probabilities
    S = len(states)     
    O = len(observations)
            
    transition_probs = np.zeros((S,S))
    emission_probs = np.zeros((O,S))
    initial_probs = np.zeros(S)
    final_probs = np.zeros(S)
    
    for line in hmm_emit_2:
        emission_probs[observations[line[1].lower()],states[line[0]]] = float(line[2])
    
    for line in hmm_trans_2:
        prec = line[0]
        follow = line[1]
        if prec == 'sentence_boundary':
            initial_probs[states[follow]] = float(line[2])
        elif follow == 'sentence_boundary':
            final_probs[states[prec]] = float(line[2])
        else:
            transition_probs[states[follow],states[prec]] = float(line[2])
        
    return states, observations, initial_probs, transition_probs, final_probs, emission_probs

def get_sentence(sent_file):
    with open(sent_file, 'r') as my_file: 
        test_sents = my_file.read().splitlines()
    
    test_sents_2 = []
    for line in test_sents:
        test_sents_2.append(line.split(' '))
    
    return test_sents_2

def scores_sent(sent, initial_probs, transition_probs, final_probs, emission_probs, observations, states):
    S = len(initial_probs)
    emission_scores = np.zeros([len(sent), S]) + logzero()
    transition_scores = np.zeros([len(sent)-1, S, S]) + logzero()
    
    initial_scores = initial_probs
    final_scores = final_probs
    
    for pos in range(len(sent)):
        word = sent[pos].lower()
        if word in observations.keys() :
            # We keep -inf for the indexes where we have no information
            ind = (emission_probs[observations[word], :] != 0)
            emission_scores[pos, ind] = emission_probs[observations[word], ind]
        else:
            emission_scores[pos, states["NONE"]] = 0
            print(word, " not in dictionnary. Uncoded NONE !")
        if pos > 0:
            transition_scores[pos-1, :, :] = transition_probs
        
    return initial_scores, transition_scores, final_scores, emission_scores

def get_string_postags(best_states_pred,inv_states):
    results = []
    for etats in best_states_pred:
        pred = ""
        start = True
        for etat in etats:
            if start:
                pred = pred + inv_states[etat]
                start = False
            else:
                pred = pred + " " + inv_states[etat]
        results.append(pred)
        
    return results

#%%
def logzero():
    return -np.inf


def safe_log(x):
    if x == 0:
        return logzero()
    return np.log(x)


def logsum_pair(logx, logy):
    """
    Return log(x+y), avoiding arithmetic underflow/overflow.

    logx: log(x)
    logy: log(y)

    Rationale:

    x + y    = e^logx + e^logy
             = e^logx (1 + e^(logy-logx))
    log(x+y) = logx + log(1 + e^(logy-logx)) (1)

    Likewise,
    log(x+y) = logy + log(1 + e^(logx-logy)) (2)

    The computation of the exponential overflows earlier and is less precise
    for big values than for small values. Due to the presence of logy-logx
    (resp. logx-logy), (1) is preferred when logx > logy and (2) is preferred
    otherwise.
    """
    if logx == logzero():
        return logy
    elif logx > logy:
        return logx + np.log1p(np.exp(logy-logx))
    else:
        return logy + np.log1p(np.exp(logx-logy))


def logsum(logv):
    """
    Return log(v[0]+v[1]+...), avoiding arithmetic underflow/overflow.
    """
    res = logzero()
    for val in logv:
        res = logsum_pair(res, val)
    return res    

# ----------
# Computes the forward trellis for a given sequence.
# Receives:
#
# Initial scores: (num_states) array
# Transition scores: (length-1, num_states, num_states) array
# Final scores: (num_states) array
# Emission scoress: (length, num_states) array
# ----------
def run_forward(initial_scores, transition_scores, final_scores, emission_scores):
    length = np.size(emission_scores, 0)  # Length of the sequence.
    num_states = np.size(initial_scores)  # Number of states.

    # Forward variables.
    forward = np.zeros([length, num_states]) + logzero()

    # Initialization.
    forward[0, :] = emission_scores[0, :] + initial_scores

    # Forward loop.
    for pos in range(1, length):
        for current_state in range(num_states):
            # Note the fact that multiplication in log domain turns a sum and sum turns a logsum
            forward[pos, current_state] = logsum(forward[pos-1, :] + transition_scores[pos-1, current_state, :])
            forward[pos, current_state] += emission_scores[pos, current_state]

    # Termination.
    log_likelihood = logsum(forward[length-1, :] + final_scores)

    return log_likelihood, forward

# ----------
# Computes the backward trellis for a given sequence.
# Receives:
#
# Initial scores: (num_states) array
# Transition scores: (length-1, num_states, num_states) array
# Final scores: (num_states) array
# Emission scoress: (length, num_states) array
# ----------
def run_backward(initial_scores, transition_scores, final_scores, emission_scores):
    length = np.size(emission_scores, 0)  # Length of the sequence.
    num_states = np.size(initial_scores)  # Number of states.

    # Backward variables.
    backward = np.zeros([length, num_states]) + logzero()

    # Initialization.
    backward[length-1, :] = final_scores

    # Backward loop.
    for pos in range(length-2, -1, -1):
        for current_state in range(num_states):
            backward[pos, current_state] = \
                logsum(backward[pos+1, :] +
                       transition_scores[pos, :, current_state] +
                       emission_scores[pos+1, :])

    # Termination.
    log_likelihood = logsum(backward[0, :] + initial_scores + emission_scores[0, :])

    return log_likelihood, backward

def compute_posteriors(initial_scores, transition_scores,
                       final_scores, emission_scores):
    """Compute the state and transition posteriors:
    - The state posteriors are the probability of each state
    occurring at each position given the sequence of observations.
    - The transition posteriors are the joint probability of two states
    in consecutive positions given the sequence of observations.
    Both quantities are computed via the forward-backward algorithm."""

    length = np.size(emission_scores, 0)  # Length of the sequence.
    num_states = np.size(emission_scores, 1)  # Number of states.

    # Run the forward algorithm.
    log_likelihood, forward = run_forward(initial_scores,
                                                       transition_scores,
                                                       final_scores,
                                                       emission_scores)

    # Run the backward algorithm.
    log_likelihood, backward = run_backward(initial_scores,
                                                         transition_scores,
                                                         final_scores,
                                                         emission_scores)

    # Multiply the forward and backward variables and divide by the
    # likelihood to obtain the state posteriors (sum/subtract in log-space).
    # Note that log_likelihood is just a scalar whereas forward, backward
    # are matrices. Python is smart enough to replicate log_likelihood
    # to form a matrix of the right size. This is called broadcasting.
    state_posteriors = np.zeros([length, num_states])  # State posteriors.
    for pos in range(length):
        state_posteriors[pos, :] = forward[pos, :] + backward[pos, :]
        state_posteriors[pos, :] -= log_likelihood

    # Use the forward and backward variables along with the transition
    # and emission scores to obtain the transition posteriors.
    transition_posteriors = np.zeros([length-1, num_states, num_states])
    for pos in range(length-1):
        for prev_state in range(num_states):
            for state in range(num_states):
                transition_posteriors[pos, state, prev_state] = \
                    forward[pos, prev_state] + \
                    transition_scores[pos, state, prev_state] + \
                    emission_scores[pos+1, state] + \
                    backward[pos+1, state]
                transition_posteriors[pos, state, prev_state] -= log_likelihood

    state_posteriors = np.exp(state_posteriors)
    transition_posteriors = np.exp(transition_posteriors)

    return state_posteriors, transition_posteriors, log_likelihood

def run_viterbi(initial_scores, transition_scores, final_scores, emission_scores):

    length = np.size(emission_scores, 0)  # Length of the sequence.
    num_states = np.size(initial_scores)  # Number of states.

    # Variables storing the Viterbi scores.
    viterbi_scores = np.zeros([length, num_states]) + logzero()

    # Variables storing the paths to backtrack.
    viterbi_paths = -np.ones([length, num_states], dtype=int)

    # Most likely sequence.
    best_path = -np.ones(length, dtype=int)
    
    # Initialization.
    viterbi_scores[0, :] = emission_scores[0, :] + initial_scores
    
    # Viterbi & Backtrack loop.
    for pos in range(1,length):
        for current_state in range(num_states):
            viterbi_scores[pos, current_state] = np.max(viterbi_scores[pos-1, :] + transition_scores[pos-1, current_state, :])
            viterbi_scores[pos, current_state] += emission_scores[pos, current_state]
            viterbi_paths[pos, current_state] = np.argmax(viterbi_scores[pos-1, :] + transition_scores[pos-1, current_state, :])
    
    # Backtrack
    best_path[length-1] = np.argmax(viterbi_scores[length-1, :] + final_scores)
    for pos in range(length-2,-1,-1):
        best_path[pos] = viterbi_paths[pos+1,best_path[pos+1]]

    return best_path, np.max(viterbi_scores[length-1, :] + final_scores)

#%%
if __name__ == '__main__':
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--transition", dest="transition", default="hmm_trans")
    opt_parser.add_option("--emission", dest="emission", default = "hmm_emits")
    opt_parser.add_option("--data", dest="data", default="dev_sents")
    opt_parser.add_option("--output", dest="output", default="candidate-postags")
    opt_parser.add_option("--viterbi", dest="viterbi", default="Y")

    (options, args) = opt_parser.parse_args()
    options = vars(options)

    print("POS-tags options:")
    for opt in options:
        print("  %-12s: %s" , (opt, options[opt]))
    print("")
    
    states, observations, initial_probs, transition_probs, final_probs, emission_probs = get_transition_emission(options['transition'],options['emission'])
    sentences = get_sentence(options['data'])
    viterbi = (options['viterbi'] == "Y")
    
    best_states_pred = []    
    if viterbi:
        for sent in sentences:
            initial_scores, transition_scores, final_scores, emission_scores = scores_sent(sent, initial_probs, transition_probs, final_probs, emission_probs, observations, states)
            best_states, _ = run_viterbi(initial_scores, transition_scores, final_scores, emission_scores)
            best_states_pred.append(best_states)
    else:
        for sent in sentences:
            initial_scores, transition_scores, final_scores, emission_scores = scores_sent(sent, initial_probs, transition_probs, final_probs, emission_probs, observations, states)
            state_posteriors, _, _ = compute_posteriors(initial_scores,
                                                             transition_scores,
                                                             final_scores,
                                                             emission_scores)
            best_states = np.argmax(state_posteriors, axis=1)
            best_states_pred.append(best_states)
    
    inv_states = {v: k for k, v in states.iteritems()}
    results = get_string_postags(best_states_pred,inv_states)            
    
    with open(options['output'], 'w') as my_file:
        for line in results:
            my_file.write(line+'\n')
    