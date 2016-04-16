import json
import random

import pandas as pd

topics = ['stay', 'leave']
sentiments = [0, 4]  # 0 for negative, 4 for positive


# read DataFrame containing tweet location info, topic and sentiment toward that topic
def read_tweets():
    data_list = ['../data/tweets_line/2016-04-' + str(x) + '.json' for x in
                 ['03', '04', '05', '06', '07', '08', '09', '10', '11', '12']]
    # data_list = ['../data/tweets_line/2016-04-' + str(x) + '.json' for x in ['10', '11']]
    tweet_list = []

    for day_file in data_list:
        with open(day_file, 'r') as f:
            print 'processing file ' + day_file
            for line in f:
                tweet = json.loads(line, encoding='latin-1')
                # filtering tweets with some location data included
                if tweet['place'] or tweet['user']['location']:
                    tweet_list.append({
                        'place': tweet['place'],
                        'userLocation': tweet['user']['location'],
                        # TODO add columns for stay/leave and sentiment (pos/neg) once Jonny creates them.
                        # TODO For now, assigns random values to these fields.
                        'topic': random.choice(topics),
                        'sentiment': random.choice(sentiments)
                    })

    df = pd.DataFrame(tweet_list)
    print len(df.index)
    return df


def map_tweet(tweet, location_coord_dict):
    try:
        bounding_box = tweet['place']['boundingBoxCoordinates'][0][0]
        coordinates = [bounding_box['latitude'], bounding_box['longitude']]
    except:
        # filter out generic locations, even if they have coordinates in the geolocation dict.
        if tweet['userLocation'] in ['United Kingdom', 'UK', 'England', 'London']:
            coordinates = None
        else:
            coordinates = location_coord_dict.get(tweet['userLocation'], None)

    return {
        'coordinates': coordinates,
        'topic': tweet['topic'],
        'sentiment': tweet['sentiment']
    }


def map_tweets_to_coordinates(df, geolocation_dict):
    coordinate_labeled_tweets = []
    row_count = len(df.index)
    for i, tweet in df.iterrows():
        if i % 10000 == 0:
            print 'Attached coordinates to %d tweets out of %d' % (i, row_count)
        coordinate_labeled_tweets.append(map_tweet(tweet, geolocation_dict))

    coordinate_labeled_tweets = filter(is_legit_tweet, coordinate_labeled_tweets)
    return coordinate_labeled_tweets


def is_legit_tweet(coord_tweet):
    return coord_tweet['coordinates'] != None and -12 < coord_tweet['coordinates'][1] < 2.5


def is_pro_leave(tweet):
    # positive towards 'leave'
    if tweet['topic'] == 'leave' and tweet['sentiment'] == 4:
        return True
    # negative towards 'stay'
    if tweet['topic'] == 'stay' and tweet['sentiment'] == 0:
        return True
    # otherwise, it's pro stay
    return False


def gen_geo_data(tweets_with_locations):
    geo = {
        'type': 'FeatureCollection',
        'features': []
    }

    for tweet in tweets_with_locations:
        geo_json_feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [
                    # coordinates are reversed for some reason - Leaflet JS needs them like this
                    tweet['coordinates'][1],
                    tweet['coordinates'][0]
                ]
            },
            'properties': {
                'text': '',
                'created_at': '',
                'vote': 'leave' if is_pro_leave(tweet) else 'remain'
            }
        }

        geo['features'].append(geo_json_feature)

    return geo


# now to write the JSON file for the map.

tweet_df = read_tweets()
with open('location_dict.json', 'r') as fin:
    tweet_location_dict = json.load(fin, encoding='latin-1')
tweets_with_locations = map_tweets_to_coordinates(tweet_df, tweet_location_dict)
geo_data = gen_geo_data(tweets_with_locations)

# Save geo data
with open('geo_data.json', 'w') as fout:
    fout.write(json.dumps(geo_data, indent=4))
