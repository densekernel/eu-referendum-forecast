import operator 
import json
from collections import Counter
from nltk.corpus import stopwords
import string
from preprocess import preprocess
from nltk import bigrams, trigrams

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['RT', 'rt', 'via', 'http', 'https', ':/']

print "punctuation", punctuation
 
fname = 'brexit.json'
with open(fname, 'r') as f:
    # freq counters
    count_all = Counter()
    count_stop = Counter()
    count_single = Counter()
    count_hash = Counter()
    count_term = Counter()
    # n-gram counters
    stop_bigram = Counter()
    term_trigram = Counter()
    for line in f:
        tweet = json.loads(line)

        tweet['text'] = tweet['text'].encode('utf-8').decode('unicode_escape').encode('ascii','ignore')
        # Create a list with all the terms
        terms_all = [term for term in preprocess(tweet['text'])]
        count_all.update(terms_all)
        # Create a list with all the terms (no stop words)
        terms_stop = [term for term in preprocess(tweet['text']) if term not in stop]
        count_stop.update(terms_stop)
        # Count terms only once, equivalent to Document Frequency
        terms_single = set(terms_all)
        count_single.update(terms_single)
        # Count hashtags only
        terms_hash = [term for term in preprocess(tweet['text']) 
                      if term.startswith('#')]
        count_hash.update(terms_hash)
        # Count terms only (no hashtags, no mentions)
        terms_only = [term for term in preprocess(tweet['text']) 
                      if term not in stop and
                      not term.startswith(('#', '@'))] 
                      # mind the ((double brackets))
                      # startswith() takes a tuple (not a list) if 
                      # we pass a list of inputs
        count_term.update(terms_only)
        # n-grams
        terms_bigram = bigrams(terms_stop)
        stop_bigram.update(terms_bigram)
        terms_trigram = trigrams(terms_all)
        term_trigram.update(terms_trigram)
        
    # Print the first 5 most frequent words
    print("Term frequencies")
    print("All: ", count_all.most_common(10))
    print("Stop: ", count_stop.most_common(10))
    print("Single: ", count_single.most_common(10))
    print("Hash: ", count_hash.most_common(10))
    print("Term only: ", count_term.most_common(10))
    print("N-gram frequencies")
    print("Stop bigrams: ", stop_bigram.most_common(10))
    print("Term trigrams: ", term_trigram.most_common(10))

    
