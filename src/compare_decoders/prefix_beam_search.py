import re
import numpy as np

class Table:
    '''
    Table class to store all of the probability values for each prefix (potential words) while performing prefix beam search.
    This allows me us more flexibility in computing new/old Beams instead of only storing the current beam probabilities
    '''
    def __init__(self):
        self.prob_table = {}
        
    def get(self, timestep, prefix):
        if timestep not in self.prob_table:
            return 0

        elif prefix not in self.prob_table[timestep]:
            return 0

        return self.prob_table[timestep][prefix]
    
    def set(self, timestep, prefix, prob):
        if timestep not in self.prob_table:
            self.prob_table[timestep] = {}          # initialize dict for a new timestep
        self.prob_table[timestep][prefix] = prob    # set prob value for the prefix at the current timestep
    
def get_num_words(txt):
    '''
    function to count words in a text. The seperator can be space or end_token
    -txt: input string
    
    -return: (int) number of words
    '''
    words = re.findall(r'\w+[\s|>]', txt)
    return len(words) + 1

def get_most_probable_beams(Prob_blank, Prob_not_blank, timestep, beamWidth, beta):
    '''
    function to get beamWidth highest probability beams from the blank and non blank probabilties
    stored for a particular timestep
    -Prob_blank: Table for the prefix ending in a blank at each timestep, eg. the prefix is a full word
    -Prob_not_blank: Table for the prefix not ending in a blank at each timestep, eg. the prefix is not a full word
    -timestep:
    -beamWidth: threshold for returning the beamWidth highest probability beams
    -beta: language model compensation (comes from prefix search args)
    
    -return: list of beamWidth highest probability beams
    '''
    prefix_list = []
    prob_map = {}

    if timestep in Prob_blank.prob_table:    
        for prefix in list(Prob_blank.prob_table[timestep].keys()):
            prefix_list.append(prefix)
            prob = Prob_blank.get(timestep, prefix) + Prob_not_blank.get(timestep, prefix)
            # word insertion factor based on the LM to prevent bias towards shorter length predictions having higher probability
            # beta is 0 for lm=None so the get_num_words function has no weight on the probability
            # TODO is beta = 0 actually true for no language model? longer word sequences "shouldn't" be penalized without it
            prob_map[prefix] = prob * get_num_words(prefix) ** beta

    if timestep in Prob_not_blank.prob_table:
        for prefix in list(Prob_not_blank.prob_table[timestep].keys()):
            if prefix not in prefix_list:        # don't recompute probability of an existing beam/prefix from above
                prefix_list.append(prefix)
                prob = Prob_blank.get(timestep, prefix) + Prob_not_blank.get(timestep, prefix)
                # word insertion factor based on the LM to prevent bias towards shorter length predictions having higher probability
                # beta is 0 for lm=None so the get_num_words function has no weight on the probability
                # TODO is beta = 0 actually true for no language model? longer word sequences "shouldn't" be penalized without it
                prob_map[prefix] = prob * get_num_words(prefix) ** beta
    prefix_list = sorted(prefix_list, key=lambda l: prob_map[l], reverse=True)
    return prefix_list[:beamWidth]

# def best_beam_search(network_matrix_output, alphabet, space_token, end_token, blank_token):
    # "Greedy" Search
    # Uses argmax of the probabilities at each time point.
    # Can either leave or remove all duplicate char sets so that AAAPPPLLLLEEE can become either apple or aple

def prefix_beam_search(network_matrix_output, alphabet, space_token, end_token, blank_token, lm, beamWidth=3, alpha=0.30, beta=5, prune=0.001):
    '''
    function to perform prefix beam search on output network_matrix_output matrix and return the best string
    -network_matrix_output: output matrix
    -alphabet: list of strings in the order their probabilties are present in network_matrix_output output
    -space_token: string representing space token
    -end_token: string representing end token
    -blank_token: string representing blank token
    -lm: function to calculate language model probability of given string
    -beamWidth: threshold for selecting the beamWidth best beams at each timestep
    -alpha: language model weight  0 if lm is None
    -beta: language model sequence length compensation  0 if lm is None
    -pruning threshold: threshold on the output matrix probability of a character. 
        If the probability of a character is less than this threshold, do not extend the prefix with it
    
    -return: best beam string
    '''
    # TODO: figure out why this is occurring or if this is normal in other algorithms
    # pad beginning with zeros so that the first character isn't cut off with a 0 prior probability
    zero_pad = np.zeros((network_matrix_output.shape[0] + 1, network_matrix_output.shape[1]))
    zero_pad[1:, :] = network_matrix_output
    network_matrix_output = zero_pad

    
    total_timesteps = network_matrix_output.shape[0]
    # num_class_predictions = network_matrix_output.shape[1]
    # print('total_timesteps: ', total_timesteps)
    # print('num_class_predictions: ', num_class_predictions)
    # print('len(alphabet): ', len(alphabet))

    # Initialization
    null_token = ''
    Prob_blank, Prob_not_blank = Table(), Table()
    Prob_blank.set(0, null_token, 1)
    Prob_not_blank.set(0, null_token, 0)
    prefix_list = [null_token] # initialize Beam List

    if (lm is None):
        alpha = 0  # make weight of language model probability 0
        # TODO is beta = 0 actually true for no language model? longer word sequences "shouldn't" be penalized without it
        beta = 0  # make weight of num words probability 0
        lm = lambda prefix : 1 # just return 1

    # calculate probabilities for every timestep
    for timestep in range(total_timesteps):
        # only evaluate characters where their predicted probabilities are above 0.001
        probable_characters = [(i, alphabet[i]) for i in np.where(network_matrix_output[timestep] > prune)[0]]
        for prefix in prefix_list:  # for every beam in the beam list, extend each and evaluate their children
            prob_prev_blank = Prob_blank.get(timestep - 1, prefix)
            prob_prev_not_blank = Prob_not_blank.get(timestep - 1, prefix)

            if len(prefix) > 0 and prefix[-1] == end_token:
                Prob_blank.set(timestep, prefix, prob_prev_blank)
                Prob_not_blank.set(timestep, prefix, prob_prev_not_blank)
                continue

            for character_index, character in probable_characters:
                new_char_confidence = network_matrix_output[timestep][character_index]    # NN confidence of the new character
                blank_char_confidence = network_matrix_output[timestep][-1]

                # if we are not sure what this timestep sounds like yet and we use a blank
                if character == blank_token:
                    # current += (our confidence * prob_prev_blank) + prob_prev_not_blank
                    value = Prob_blank.get(timestep, prefix) + (new_char_confidence * prob_prev_blank) + prob_prev_not_blank
                    Prob_blank.set(timestep, prefix, value)

                else:
                    prefix_extended = prefix + character
                    if prefix and prefix[-1] == character:  # new char is the same as the last char and not empty string
                        value = Prob_not_blank.get(timestep, prefix_extended) + (new_char_confidence * prob_prev_blank)
                        Prob_not_blank.set(timestep, prefix_extended, value)
                        value = Prob_not_blank.get(timestep, prefix) + (new_char_confidence * prob_prev_not_blank)
                        Prob_not_blank.set(timestep, prefix, value)

                    elif prefix.replace(space_token, '') and character in (space_token, end_token):  # end of a word (need to check language model)
                        # check probability of that word based on the trained language model: probably not a space/end token if it is a misspelled word
                        lm_prob = lm(prefix_extended.strip(space_token + end_token)) ** alpha
                        # might not need to strip the space token for Fairseq LM model because it counted _ as start word token
                        # might need to take softmax or log_softmax of the lm() function output depending on lm model; I think it has log_prob stored for fairseq
                        value = Prob_not_blank.get(timestep, prefix_extended) + ((new_char_confidence * lm_prob) * (prob_prev_blank + prob_prev_not_blank))
                        Prob_not_blank.set(timestep, prefix_extended, value)
                    else:   # if not end of a word, just add the new probability conditioned on the last probability again
                        value = Prob_not_blank.get(timestep, prefix_extended) + (new_char_confidence * (prob_prev_blank + prob_prev_not_blank))
                        Prob_not_blank.set(timestep, prefix_extended, value)

                    if prefix_extended not in prefix_list:  # update new beam probabilities
                        value = Prob_blank.get(timestep, prefix_extended) + (blank_char_confidence * (Prob_blank.get(timestep - 1, prefix_extended) + Prob_not_blank.get(timestep - 1, prefix_extended)))
                        Prob_blank.set(timestep, prefix_extended, value)        # update new beam probability for ending right now (blank)
                        value = Prob_not_blank.get(timestep, prefix_extended) + (new_char_confidence * Prob_not_blank.get(timestep - 1, prefix_extended))
                        Prob_not_blank.set(timestep, prefix_extended, value)    # update new beam probability for not ending (not blank)

        prefix_list = get_most_probable_beams(Prob_blank, Prob_not_blank, timestep, beamWidth, beta)  # prune all beams to only the beamWidth best beams

    # Output
    return prefix_list[0].strip(end_token)  # return just the best beam/prefix without the end token
