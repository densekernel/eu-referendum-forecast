import pandas as pd
import operator 
import json
from collections import Counter
from nltk.corpus import stopwords
import string
from preprocess import preprocess
from nltk import bigrams, trigrams
import matplotlib.pyplot as plt

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['RT', 'rt', 'via', 'http', 'https', ':/']

print "punctuation", punctuation
 
fname = 'brexit.json'

dates_stay = []
dates_leave = []

with open(fname, 'r') as f:
  # f is the file pointer to the JSON data set
  for line in f:
      tweet = json.loads(line)
      # let's focus on hashtags only at the moment
      terms_all = [term for term in preprocess(tweet['text'])]
      # terms_hash = [term for term in preprocess(tweet['text']) if term.startswith('#')]
      # track when the hashtag is mentioned
      if 'stay' in terms_all:
          dates_stay.append(tweet['created_at'])
      if 'leave' in terms_all:
          dates_leave.append(tweet['created_at'])
   
  # a list of "1" to count the hashtags
  ones_stay = [1]*len(dates_stay)
  ones_leave = [1]*len(dates_leave)
  # the index of the series
  idx_stay = pd.DatetimeIndex(dates_stay)
  idx_leave = pd.DatetimeIndex(dates_leave)
  # the actual series (at series of 1s for the moment)
  series_stay = pd.Series(ones_stay, index=idx_stay)
  series_leave = pd.Series(ones_leave, index=idx_leave)
   
  # Resampling / bucketing
  per_minute_stay = series_stay.resample('5Min', how='sum').fillna(0)
  per_minute_leave = series_leave.resample('5Min', how='sum').fillna(0)

  fig = plt.figure()
  plt.plot(per_minute_stay, label='stay')
  plt.plot(per_minute_leave, label='leave')
  plt.legend()
  plt.title("Twitter Mentions of 'stay' vs. 'leave'. (5-minutely resample)")
  plt.show()

  per_minute_stay_cumsum = per_minute_stay.cumsum()
  per_minute_leave_cumsum = per_minute_leave.cumsum()

  fig = plt.figure()
  plt.plot(per_minute_stay_cumsum, label='stay')
  plt.plot(per_minute_leave_cumsum, label='leave')
  plt.legend()
  plt.title("Cumulative Twitter Mentions of 'stay' vs. 'leave'. (5-minutely resample)")
  plt.show()

