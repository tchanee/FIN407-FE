import sys
from typing import Dict

import pandas as pd

def annotate(read_from: str, write_to: str, label_colname: int, id2label: Dict[int, str]):
    
    print(f"Labeling the data in {read_from}")
    print(f"The labeled data will be written to {write_to} in column {label_colname}")
    
    tweets = pd.read_csv(read_from, index_col=0)
    labels = []
    for tweet in tweets.itertuples():
        print("ENTRY:")
        print("TEXT= ", tweet.text)
        
        valid_label_id = False
        while not(valid_label_id):
            try:
                label_id = int(input("LABEL= "))
            except Exception:
                print("** ERROR: Invalid label ID, which should be 0 or 1 or 2. Enter a new label...")
                continue
            valid_label_id = set(id2label.keys()).__contains__(label_id)
            if not(valid_label_id):
                print("** Error: Invalid label ID, which should be 0 or 1 or 2. Repeat a new label...")
                continue
        
        print("CHOSEN LABEL: ", id2label[label_id])
        labels.append(id2label[label_id])
        
    tweets[label_colname] = labels
    tweets.to_csv(write_to)
    
    print("done!")

if __name__ == "__main__":
    arg2val = {}
    for arg in sys.argv[1:]:
        arg, val = arg.split("=")
        arg2val[arg] = val
    
    read_from = arg2val.get("--read_from", "../data/test_tweets.csv")
    write_to  = arg2val.get("--write_to", "../data/test_tweets_labeled.csv")
    annotator = arg2val.get("--annotator", "unknown")
    
    id2label = {0: "negative", 1: "positive", 2: "neutral"}
    
    annotate(read_from, write_to, f"label_{annotator}", id2label)