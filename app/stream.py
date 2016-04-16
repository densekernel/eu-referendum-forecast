import os
import tweepy

from dotenv import load_dotenv
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

class MyListener(StreamListener):
 
    def on_data(self, data):
        try:
            with open('brexit.json', 'a') as f:
                f.write(data)
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True
 
    def on_error(self, status):
        print(status)
        return True

load_dotenv('.env')
consumer_key =  os.environ.get('consumer_key')
consumer_secret =  os.environ.get('consumer_secret')
access_token =  os.environ.get('access_token')
access_secret =  os.environ.get('access_secret')

print consumer_key, consumer_secret, access_token, access_secret

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

track_stay = ['#yes2eu', '#yestoeu', '#betteroffin', '#votein', '#ukineu', '#bremain', '#strongerin', '#leadnotleave', '#voteremain']

track_leave = ['#brexit', '#no2eu', '#notoeu', '#betteroffout', '#voteout', '#britainout', '#leaveeu', '#loveeuropeleaveeu', '#voteleave', '#beleave']

brexit_track = ['#brexit', '#no2eu', '#notoeu', '#betteroffout', '#voteout', '#britainout', '#leaveeu', '#loveeuropeleaveeu', '#voteleave', '#beleave', '#yes2eu', '#yestoeu', '#betteroffin', '#votein', '#ukineu', '#bremain', '#strongerin', '#leadnotleave', '#voteremain']

# start twitter stream (intermittent collection beginngin on 20:40 30/03/2016)
twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=brexit_track)

