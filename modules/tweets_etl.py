import pandas as pd

##############################################
# Transformations on tweets
##############################################

def process_for_bsts(tweets: str):
    tweets = tweets.rename(columns={"day": "date"})
    tweets = tweets.convert_dtypes()
    tweets["date"] = pd.to_datetime(tweets["date"], format="%Y-%m-%d")
    return tweets

