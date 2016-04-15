import json
from urllib import urlencode
from urllib2 import urlopen
from time import sleep

import pandas as pd

__author__ = 'Gabriel'


# read DataFrame containing tweet location info, topic and sentiment toward that topic
def read_tweet_locations():
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
                if tweet['place'] or tweet['user']['isGeoEnabled'] is True:
                    tweet_list.append({
                        'place': tweet['place'],
                        'userLocation': tweet['user']['location'],
                    })

    df = pd.DataFrame(tweet_list)
    print len(df.index)
    return df


# Get coordinates of address using Google Geocoding API
def get_coords(user_location):
    if user_location is None:
        return None

    try:
        url = 'http://nominatim.openstreetmap.org/search?' + urlencode({'q': user_location, 'format': 'json'})
        location_json = json.loads(urlopen(url).read())
        if len(location_json) == 0:
            return None

        return [
            float(location_json[0]['lat']),  # latitude
            float(location_json[0]['lon'])  # longitude
        ]
    except:
        return None


# generate dictionary from user.location strings to geolocation(latitude, longitude)
tweet_df = read_tweet_locations()

# user location -> [latitude, longitude]
location_coord_dict = {}

# get all distinct user locations. Ignore those of records where coordinates are specified
# noinspection PyComparisonWithNone
distinct_user_locations = pd.Series(tweet_df[tweet_df['userLocation'] != None]['userLocation']).unique()

loc_count = len(distinct_user_locations)
print loc_count
i = 0

for location in distinct_user_locations:
    coords = get_coords(location)
    sleep(1)
    location_coord_dict[location] = coords

    i += 1
    if i % 10 == 0:
        print 'Processed %d locations out of %d' % (i, loc_count)

with open('location_dict.json', 'w+') as fout:
    fout.write(json.dumps(location_coord_dict, indent=4))
