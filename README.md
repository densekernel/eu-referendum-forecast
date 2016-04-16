# Deep Referendum

## Synopsis

This is the UCL IRDM 2016 group project for

## Members

1. Jonny Manfield
1. Vlad Kolesnyk
1. Gabriel Bila

## Installation

### BrexitTcf 

BrexitTcf is responsible for downloading previous #Brexit tweets. It uses the Twitter search API to download tweets between 2 dates.

BrexitTcf is written in Scala and uses Gradle for dependency management and build automation.

To run BrexitTcf search API, make sure you have the following installed first:

1. JDK 8 (make sure JAVA_HOME points to the right jdk directory)
1. Scala (latest version)
1. Gradle (latest version)

From the BrexitTcf directory, run ```gradle installDist``` to generate the project JARs and launch scripts. They will be placed under ```BrexitTcf/build/install/BrexitTcf```

Then, to launch the Brexit search, ```cd``` to ```build/install/BrexitTcf/bin``` and run ```./BrexitTcf <start-date> <end-date>```. For example ```./BrexitTcf 2016-04-10 2016-04-12```.

This will query the Twitter API for #Brexit-related tweets. The tweets will be output in the current directory. One separate JSON file will be created for each day.

This only works for tweets 1 week old or less. Twitter API doesn't return historical tweets older than this.

## Reference

<!--List all Python scripts and other components here. Use alphabetical order to keep things nice and clean.--> 

* ```app/cooccurence.py```:

* ```app/geo.py``` Generates tweet coordinates into ```geo_data.json```, the file used by ```map.html``` for visualising tweets over the map of the UK.

* ```app/map.html``` Displays tweet heat map over the UK: blue represents tweets for staying in the EU, red represents tweets for leaving the EU. Selecting 'Overlap heatmaps' plots both 
'leave' and 'stay' heat maps on top of each other. Blue will then represent majority pro-stay areas, red - majority pro-leave, and purple - highly-contested areas. 

   First, you need to generate data for the heat maps. To do so, run ```python geo.py``` first.

* ```app/location_coord_dict.py``` assigns geolocation to tweets based on their ```user.location``` value. Queries OSM Nominatim to convert the user's address to a pair of coordinates.
  The output is a dictionary in the format ```{address -> [lat, long]}```. This dictionary is stored in ```app/location_dict.json```

* ```app/stream.py```: __TCF Streaming API__. To run, enter ```python stream.py```

   TCF has two major components. The first one is Streaming - listening on Twitter for incoming tweets for the #Brexit track.

* ```app/termfreq.py```: Run ```python termfreq.py``` to list term frequency metrics for tweets.

* ```BrexitTcf``` __TCF Search API__. See the corresponding __Installation__ section on how to install and run.
