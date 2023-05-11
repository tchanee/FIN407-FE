import pandas
import pandas as pd
import datetime as dt

from typing import Dict, List

##############################################
# Tweets ETL for BSTS
##############################################

def process_tweets_dict(tweets_dict: Dict[str, Dict[str, pandas.DataFrame]], most_popular: int =-1) -> pandas.DataFrame:
    assert most_popular != 0, "most_popular must be different from 0."
    
    new_tweets_dict = {}
    
    for partition_name, sentiment_dict in tweets_dict.items():
        new_tweets_dict[partition_name] = {}
        for sentiment, tweets in sentiment_dict.items():
            assert len(tweets.tweetbert_sentiment.unique()) > 0, "BERTweet sentiment must be unique, since the dictionnary is partitionend by BERTweet sentiment."
            
            tweets = tweets.sort_values(by="like_count", ascending=False)
            
            if most_popular > 0:
                tweets = tweets.head(most_popular)
            
            tweets = tweets.astype({"like_count": int, "text": str, "id": str})
            tweets["date"] = pd.to_datetime(tweets["day"], format="%Y-%m-%d")
            
            tweets.drop(columns=[
                'created_at', 'hashtags', 'mentions', 'cashtags', 'is_reply', 
                'retweet_count', 'reply_count', 'quote_count', 'impression_count',
                'text_length', 'day', 'processed_text', 'month',
                'finbert_sentiment'
                ], inplace=True)
            tweets.rename(columns={"like_count": "likes", "tweetbert_sentiment": "bertweet", "vader_sentiment": "vader"}, inplace=True)
            
            new_tweets_dict[partition_name][sentiment] = tweets

    return new_tweets_dict

def extract_tweets_dict_texts(tweets_dict: Dict[str, Dict[str, pandas.DataFrame]]) -> Dict[str, Dict[str, List[str]]]:
    result = {}
    
    for partition_name, sentiment_dict in tweets_dict.items():
        result[partition_name] = {}
        for sentiment, tweets in sentiment_dict.items():
            result[partition_name][sentiment] = list(zip(tweets.id.values.tolist(), tweets.text.values.tolist()))
    
    return result

def extract_extrema_dates(tweets_dict: Dict[str, Dict[str, pandas.DataFrame]]) -> Dict[str, Dict[str, dt.datetime]]:
    result = {}
    
    for asset_name, sentiment_dict in tweets_dict.items():
        result[asset_name] = {}
        for sentiment, tweets in sentiment_dict.items():
            result[asset_name][sentiment] = (tweets.date.min(), tweets.date.max())
    
    return result

def extract_tweet(tweets_dict: Dict[str, Dict[str, pandas.DataFrame]], tweet_id: str):
    for _, sentiment_dict in tweets_dict.items():
        for _, tweets in sentiment_dict.items():
            tweet = tweets[tweets.id == tweet_id]
            if len(tweet) > 0:
                return tweet
    return None
