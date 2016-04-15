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
                if tweet['place']:
                    tweet_list.append({
                        'place': tweet['place'],
                        # TODO add column for user.location when integrating with geolocation dictionary
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
        coordinates = tweet['place']['boundingBoxCoordinates'][0][0]
    except KeyError:
        coordinates = location_coord_dict.get(tweet['userLocation'], None)

    return {
        'coordinates': coordinates,
        'topic': tweet['topic'],
        'sentiment': tweet['sentiment']
    }


def map_tweets_to_coordinates(df):
    # TODO uncomment when loading geolocation dict
    # location_coord_dict = {}

    # distinct_user_locations = pd.Series(df[df['userLocation'] is not None]['userLocation']).unique()
    # for user_location in distinct_user_locations:
    #     user_coords = get_coords(user_location)
    #     location_coord_dict[user_location] = user_coords

    coordinate_labeled_tweets = []
    for i, tweet in df.iterrows():
        coordinate_labeled_tweets.append(map_tweet(tweet, {}))

    coordinate_labeled_tweets = filter(lambda tw: tw['coordinates'] != None, coordinate_labeled_tweets)
    return coordinate_labeled_tweets


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
                    tweet['coordinates']['longitude'],
                    tweet['coordinates']['latitude']
                ]
            },
            'properties': {
                'text': '',
                'created_at': ''
            }
        }

        geo['features'].append(geo_json_feature)

    return geo


# now to write the JSON file for the map.

tweet_df = read_tweets()
tweets_with_locations = map_tweets_to_coordinates(tweet_df)
geo_data = gen_geo_data(tweets_with_locations)

# Save geo data
with open('geo_data.json', 'w') as fout:
    fout.write(json.dumps(geo_data, indent=4))
