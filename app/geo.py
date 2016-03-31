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

fname = 'brexit.json'

# Tweets are stored in "fname"
with open(fname, 'r') as f:
    geo_data = {
        "type": "FeatureCollection",
        "features": []
    }
    for line in f:
        tweet = json.loads(line)
        # print tweet['coordinates'] 
        if tweet['coordinates']:
            print "tweettweet"
            try:
                geo_json_feature = {
                    "type": "Feature",
                    "geometry": tweet['coordinates'],
                    "properties": {
                        "text": tweet['text'],
                        "created_at": tweet['created_at']
                    }
                }
                geo_data['features'].append(geo_json_feature)
            except:
                print "Unexpected error:", sys.exc_info()[0]


# Save geo data
with open('geo_data.json', 'w') as fout:
    fout.write(json.dumps(geo_data, indent=4))
