import json
import pickle

topics = ['stay', 'leave']
sentiments = [0, 4]  # 0 for negative, 4 for positive


def read_tweets():
    print "Reading tweets from pkl file"
    with open('../data/brexit.pkl', 'rb') as f:
        df = pickle.load(f)
    print "Got the tweets"
    return df


def tweet_coords(tweet, location_coord_dict):
    try:
        bounding_box = tweet['place']['boundingBoxCoordinates'][0][0]
        coordinates = [bounding_box['latitude'], bounding_box['longitude']]
    except:
        # filter out generic locations, even if they have coordinates in the geolocation dict.
        if tweet['user']['location'] in ['United Kingdom', 'UK', 'England', 'London']:
            coordinates = None
        else:
            coordinates = location_coord_dict.get(tweet['user']['location'], None)
    return coordinates


def is_pro_leave(tweet):
    # positive towards 'leave'
    if tweet['subject'] == 'leave' and tweet['sentiment'] == 'pos':
        return True
    # negative towards 'stay'
    if tweet['subject'] == 'stay' and tweet['sentiment'] == 'neg':
        return True
    # otherwise, it's pro stay or neutral
    return False


def is_pro_stay(tweet):
    # positive towards 'stay'
    if tweet['subject'] == 'stay' and tweet['sentiment'] == 'pos':
        return True
    # negative towards 'leave'
    if tweet['subject'] == 'leave' and tweet['sentiment'] == 'neg':
        return True
    # otherwise, it's pro leave or neutral
    return False


def map_tweet(tweet, location_coord_dict):
    return {
        'coordinates': tweet_coords(tweet, location_coord_dict),
        'subject': tweet['subject_strict'],
        'sentiment': tweet['sentiment'],
        'pro_leave': is_pro_leave(tweet),
        'pro_stay': is_pro_stay(tweet)
    }


def map_tweets_to_coordinates(df, geolocation_dict):
    coordinate_labeled_tweets = []
    row_count = len(df.index)
    for i, tweet in df.iterrows():
        if i % 10000 == 0:
            print 'Attached coordinates to %d tweets out of %d' % (i, row_count)
        coordinate_labeled_tweets.append(map_tweet(tweet, geolocation_dict))

    coordinate_labeled_tweets = filter(is_legit_tweet, coordinate_labeled_tweets)
    print 'tweets with coordinates: ' + str(len(coordinate_labeled_tweets))
    return coordinate_labeled_tweets


def is_legit_tweet(coord_tweet):
    return coord_tweet['coordinates'] != None and -12 < coord_tweet['coordinates'][1] < 2.5


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
                'pro_leave': tweet['pro_leave'],
                'pro_stay': tweet['pro_stay']
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
