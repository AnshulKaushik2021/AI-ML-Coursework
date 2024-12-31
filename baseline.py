"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
from collections import defaultdict, Counter
def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    word_tag_counts = defaultdict(Counter)
    overall_tag_count = Counter()
    
    for sentence in train:
        for word, tag in sentence:
            word_tag_counts[word][tag] += 1
            overall_tag_count[tag] += 1
    
    most_common_tag = {word: tags.most_common(1)[0][0] for word, tags in word_tag_counts.items()}
    default_tag = overall_tag_count.most_common(1)[0][0]
    
    tagged_sentences = []
    for sentence in test:
        tagged_sentence = [(word, most_common_tag.get(word, default_tag)) for word in sentence]
        tagged_sentences.append(tagged_sentence)
    
    return tagged_sentences