import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5  # exact setting seems to have little or no effect

def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: initial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0)  # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0))  # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0))  # {tag0:{tag1: # }}

    # Count occurrences
    tag_counts = Counter()  # Total counts of tags
    for sentence in sentences:
        prev_tag = None
        for word, tag in sentence:
            tag_counts[tag] += 1
            if prev_tag is not None:
                trans_prob[prev_tag][tag] += 1
            prev_tag = tag
            emit_prob[tag][word] += 1

    # Calculate initial probabilities
    total_sentences = len(sentences)
    for tag, count in tag_counts.items():
        init_prob[tag] = count / total_sentences

    # Calculate emission probabilities
    for tag, words in emit_prob.items():
        total_words = sum(words.values())
        for word in words:
            emit_prob[tag][word] = (words[word] + emit_epsilon) / (total_words + emit_epsilon * (len(words) + 1))
        emit_prob[tag]['UNKNOWN'] = emit_epsilon / (total_words + emit_epsilon * (len(words) + 1))

    # Calculate transition probabilities
    for tag0, next_tags in trans_prob.items():
        total_transitions = sum(next_tags.values())
        for tag1 in next_tags:
            trans_prob[tag0][tag1] = (next_tags[tag1] + epsilon_for_pt) / (total_transitions + epsilon_for_pt * (len(next_tags) + 1))

    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {}  # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {}  # This should store the tag sequence to reach each tag at column (i)

    # Handle the special case for the first column
    if i == 0:
        for tag in emit_prob:
            # Use emission probability of word for the initial step
            if word in emit_prob[tag]:
                log_prob[tag] = log(emit_prob[tag][word]) + log(epsilon_for_pt)  # Log of initial prob
            else:
                log_prob[tag] = log(emit_prob[tag]['UNKNOWN']) + log(epsilon_for_pt)
            predict_tag_seq[tag] = []
    else:
        for curr_tag in emit_prob:
            max_prob = float('-inf')
            best_prev_tag = None
            for prev_tag in prev_prob:
                trans_log_prob = log(trans_prob[prev_tag].get(curr_tag, epsilon_for_pt))
                prob = prev_prob[prev_tag] + trans_log_prob

                if prob > max_prob:
                    max_prob = prob
                    best_prev_tag = prev_tag

            log_prob[curr_tag] = max_prob + log(emit_prob[curr_tag].get(word, emit_prob[curr_tag]['UNKNOWN']))
            predict_tag_seq[curr_tag] = prev_predict_tag_seq[best_prev_tag] + [best_prev_tag]

    return log_prob, predict_tag_seq

def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)

    predicts = []
    
    for sen in range(len(test)):
        sentence = test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        
        # Initialize log probabilities for the first column
        for t in emit_prob:
            log_prob[t] = log(init_prob.get(t, epsilon_for_pt))
            predict_tag_seq[t] = []

        # Forward steps to calculate log probabilities for the sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob)

        # TODO:(III) 
        # According to the storage of probabilities and sequences, get the final prediction.
        best_final_tag = max(log_prob, key=log_prob.get)
        predicts.append(list(zip(sentence, predict_tag_seq[best_final_tag] + [best_final_tag])))

    return predicts
