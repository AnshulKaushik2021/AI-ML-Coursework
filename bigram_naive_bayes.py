# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, and (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign.
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023

"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful not to modify anything else.
"""

import reader
import math
from tqdm import tqdm
from collections import Counter


'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")


"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir, testdir, stemming, lowercase, silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigram_bayes(train_set, train_labels, dev_set, unigram_laplace=1.0, bigram_laplace=1.0, bigram_lambda=0.4, pos_prior=0.5, silently=False):
    print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior)

    # Expand on counter and setup variables from MP1
    pword_total, negword_total = 0, 0
    pbigram_total, negbigram_total = 0, 0
    punigram_count, negunigram_count = Counter(), Counter()
    pbigram_count, negbigram_count = Counter(), Counter()
    

    # Add additional vocab set compared to MP1
    uniVocab, biVocab = set(), set()
    # For loop used for getting vocabularies set up
    for i, doc in enumerate(train_set):
        
        words = doc
        unigrams = words
        list_bi = list(zip(words, words[1:]))
        label = train_labels[i]

        if label == 1:  
            punigram_count.update(unigrams)
            pword_total += len(unigrams)
            pbigram_count.update(list_bi)
            pbigram_total += len(list_bi)
        else:  
            negunigram_count.update(unigrams)
            negword_total += len(unigrams)
            negbigram_count.update(list_bi)
            negbigram_total += len(list_bi)

        uniVocab.update(unigrams)
        biVocab.update(list_bi)

    uniVocab_size = len(uniVocab)
    biVocab_size = len(biVocab)

    # Now we find the position prior and the 1 minus it
    pos_prior_log = math.log(pos_prior)
    neg_prior_log = math.log(1 - pos_prior)

    # Function inside a function to find the log probability based on given equation
    def loggy(doc, is_bigram=False):
        words = doc
        if is_bigram:
            ngrams = list(zip(words, words[1:]))  
        else:
            ngrams = words 

        pos_log_prob, neg_log_prob = pos_prior_log, neg_prior_log
        if is_bigram:
            for ngram in ngrams:
                pos_count = pbigram_count[ngram] + bigram_laplace
                pos_prob = math.log(pos_count) - math.log(pbigram_total + biVocab_size * bigram_laplace)
                pos_log_prob += pos_prob
                
                neg_count = negbigram_count[ngram] + bigram_laplace      
                neg_prob = math.log(neg_count) - math.log(negbigram_total + biVocab_size * bigram_laplace)       
                neg_log_prob += neg_prob
        else:
            for ngram in ngrams:
                pos_count = punigram_count[ngram] + unigram_laplace
                pos_prob = math.log(pos_count) - math.log(pword_total + uniVocab_size * unigram_laplace)
                pos_log_prob += pos_prob
                
                neg_count = negunigram_count[ngram] + unigram_laplace
                neg_prob = math.log(neg_count) - math.log(negword_total + uniVocab_size * unigram_laplace)
                neg_log_prob += neg_prob

        return pos_log_prob, neg_log_prob

    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        unigram_pos_log_prob, unigram_neg_log_prob = loggy(doc, is_bigram=False)
        bigram_pos_log_prob, bigram_neg_log_prob = loggy(doc, is_bigram=True)

        positive = (1 - bigram_lambda) * unigram_pos_log_prob + bigram_lambda * bigram_pos_log_prob
        negative = (1 - bigram_lambda) * unigram_neg_log_prob + bigram_lambda * bigram_neg_log_prob

        final_prediction = 1 if positive > negative else 0
        yhats.append(final_prediction)
    # Hi
    return yhats




