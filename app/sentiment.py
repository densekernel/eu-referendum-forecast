import operator 
import json
from collections import Counter
from nltk.corpus import stopwords
import string
from preprocess import preprocess
from nltk import bigrams, trigrams
from collections import defaultdict
import math
import numpy as np

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['RT', 'rt', 'via', 'http', 'https', ':/']
 
fname = 'brexit.json'
 
com = defaultdict(lambda : defaultdict(int))
count_stop_single = Counter()
n_docs = 0.0

with open(fname, 'r') as f:
    # f is the file pointer to the JSON data set
    for line in f: 
        n_docs += 1
        tweet = json.loads(line)
        tweet['text'] = tweet['text'].encode('utf-8').decode('unicode_escape').encode('ascii','ignore')
        # terms_only = [term for term in preprocess(tweet['text']) 
                      # if term not in stop 
                      # and not term.startswith(('#', '@'))]

        # Create a list with all the terms (no stop words)
        terms_stop = [term for term in preprocess(tweet['text']) if term not in stop]
        # count_stop.update(terms_stop)
        # Count terms only once, equivalent to Document Frequency
        terms_stop_single = set(terms_stop)
        count_stop_single.update(terms_stop_single)
     
        # Build co-occurrence matrix
        for i in range(len(terms_stop)-1):            
            for j in range(i+1, len(terms_stop)):
                w1, w2 = sorted([terms_stop[i], terms_stop[j]])                
                if w1 != w2:
                    com[w1][w2] += 1

p_t = {}
p_t_com = defaultdict(lambda : defaultdict(int))

for term, n in count_stop_single.items():
    p_t[term] = n / n_docs
    for t2 in com[term]:
        p_t_com[term][t2] = com[term][t2] / n_docs

positive_vocab = [
    'good', 'nice', 'great', 'awesome', 'outstanding',
    'fantastic', 'terrific', ':)', ':-)', 'like', 'love',
    # shall we also include game-specific terms?
    # 'triumph', 'triumphal', 'triumphant', 'victory', etc.
]
negative_vocab = [
    'bad', 'terrible', 'crap', 'useless', 'hate', ':(', ':-(',
    # 'defeat', etc.
]

pmi = defaultdict(lambda : defaultdict(int))
for t1 in p_t:
    for t2 in com[t1]:
        denom = p_t[t1] * p_t[t2]
        pmi[t1][t2] = np.log2(p_t_com[t1][t2] / denom)
 
semantic_orientation = {}
for term, n in p_t.items():
    positive_assoc = sum(pmi[term][tx] for tx in positive_vocab)
    negative_assoc = sum(pmi[term][tx] for tx in negative_vocab)
    semantic_orientation[term] = positive_assoc - negative_assoc

print("#brexit: %f" % semantic_orientation['#brexit'])


# com_max = []
# # For each term, look for the most common co-occurrent terms
# for t1 in com:
#     t1_max_terms = max(com[t1].items(), key=operator.itemgetter(1))[:5]
#     for t2 in t1_max_terms:
#         com_max.append(((t1, t2), com[t1][t2]))
# # Get the most frequent co-occurrences
# terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
# print(terms_max[:5])

# print("Co-occurrence for %s:" % search_word)
# print(count_search.most_common(20))

# n_docs is the total n. of tweets
