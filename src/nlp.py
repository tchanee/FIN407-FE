# author: FAROUK BOUKIL

from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import pipeline
 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import torch
import numpy as np

#################################################################
# Named Entity Recognition Models
#################################################################

class BERTWrapperForNER():
    DEFAULT_ID = "Davlan/distilbert-base-multilingual-cased-ner-hrl"
    
    def __init__(self, model_id_or_path: str = DEFAULT_ID):
        self.model_name = model_id_or_path.split("/")[-1]
        
        self.org_start_label = "B-ORG"
        self.org_middle_label = "I-ORG"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        self.model     = AutoModelForTokenClassification.from_pretrained(model_id_or_path)
        self.model.eval()
         
        self.pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
    
    def apply(self, texts):
        return self.pipeline(texts)
        
    def __call__(self, texts):
        outputs = self.apply(texts)
        
        all_entities = []
        for output in outputs:
            entities = []
            entity_name = ""
            for token_data in output:
                # if the token is at the start of an org entity, initialize a new entity name
                # if the entity name is not empty, flush it to the list before initializing a new entity name
                name_is_empty = len(entity_name) == 0
                if token_data["entity"] == self.org_start_label:
                    if not(name_is_empty):
                        entities.append(entity_name)
                    entity_name = token_data["word"].strip("##")
                # if the token is in the middle of an org entity, append it to the entity name
                elif token_data["entity"] == self.org_middle_label and not(name_is_empty):
                        entity_name += token_data["word"].strip("##")
                # if the label is not an org label, flush the entity name to the list if not empty
                else:
                    if not(name_is_empty):
                        entities.append(entity_name)
                        entity_name = ""
            
            # flush the last entity name into the list if any
            if len(entity_name) > 0:
                entities.append(entity_name)
                
            all_entities.append(entities)

        return all_entities
    
    def to(self, device):
        self.model = self.model.to(device)
        self.pipeline = self.pipeline("ner", model=self.model, tokenizer=self.tokenizer, device=device)
            
    def save_to(self, path):
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)
                
    @staticmethod
    def load_from(path):
        return BERTWrapperForNER(path)

#################################################################
# Sentiment Analysis Models
#################################################################

class BERTWrapperForSA():
    FIN_BERT_ID = "ProsusAI/finbert"
    TWEET_BERT_ID = "finiteautomata/bertweet-base-sentiment-analysis"
    BASE_BERT_ID = "DunnBC22/bert-base-uncased-Twitter_Sentiment_Analysis_v2"
    
    @staticmethod
    def __adapt_label(label):
        if label in ["positive", "POS"]:
            return "positive"
        elif label in ["negative", "NEG"]:
            return "negative"
        else:
            return "neutral"
    
    def __init__(self, model_id_or_path: str, verbalizer=None):
        # model
        self.model_name = model_id_or_path.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id_or_path)
        self.model.eval()
        # decoding
        self.amplification_threshold = 0.4
        self.sentiment_dominance_ratio = 1.5
        
        self.pos_id = self.model.config.label2id.get("positive", None)
        self.neg_id = self.model.config.label2id.get("negative", None)
        self.neu_id = self.model.config.label2id.get("neutral", None)
        if self.pos_id is None or self.neg_id is None or self.neu_id is None:
            self.pos_id = self.model.config.label2id["POS"]
            self.neg_id = self.model.config.label2id["NEG"]
            self.neu_id = self.model.config.label2id["NEU"]
        
        # verbalization
        self.verbalizer = (lambda pred_id: BERTWrapperForSA.__adapt_label(self.model.config.id2label[pred_id])) if verbalizer is None else verbalizer
    
    def verbalize(self, pred_ids):
        return [self.verbalizer(pred_id) for pred_id in pred_ids]
        
    def decode(self, probas, decoding="greedy"):
        if decoding == "greedy":
            pred_ids = probas.argmax(dim=-1).tolist()
            return pred_ids
        else:
            raise ValueError(f"Decoding method '{decoding}' not supported")
         
    def predict(self, texts, labeled=False):
        with torch.no_grad():
            batch_encoding = self.tokenizer.batch_encode_plus(texts, return_tensors="pt", truncation=True, padding=True)
            logits = self.model(**batch_encoding).logits
        probas = torch.softmax(logits, dim=-1)
        
        if labeled:
            probas = [{self.verbalizer(i):proba.item() for i, proba in enumerate(distribution)} for distribution in probas]
        
        return probas
    
    def __call__(self, texts, decoding="greedy"):
        with torch.no_grad():
            probas = self.predict(texts)
            pred_ids = self.decode(probas, decoding)
            preds = self.verbalize(pred_ids)
            return preds
    
    def to(self, device):
        self.model = self.model.to(device)
        
    def save_to(self, path):
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)
    
    @staticmethod
    def load_from(path):
        return BERTWrapperForSA(path)
  

class VaderSentimentWrapper():
    def __init__(self, verbalizer=None):
        # model
        self.model = SentimentIntensityAnalyzer()
        self.score_func = np.frompyfunc(self.model.polarity_scores, 1, 1)
        # verbalization
        self.__label2id = {"negative": 0, "neutral": 1, "positive": 2}
        self.__id2label = {v:k for k, v in self.__label2id.items()}
        self.verbalizer = (lambda pred_id:  self.__id2label[pred_id]) if verbalizer is None else verbalizer
        # decoding
        self.amplification_threshold = 0.35
        self.sentiment_dominance_ratio = 2
        
    def verbalize(self, pred_ids):
        return [self.verbalizer(pred_id) for pred_id in pred_ids]
       
    def decode(self, scores, decoding="greedy"):
        scores = torch.Tensor([list(score.values()) for score in scores])
        if decoding == "greedy":
            pred_ids = scores.argmax(dim=-1).tolist()
            return pred_ids
        else:
            raise ValueError(f"Decoding method '{decoding}' not supported")
    
    def compute_scores(self, texts):
        scores = self.score_func(np.array(texts)).tolist()
        
        clean_scores = []
        for score in scores:
            del score["compound"]
            clean_scores.append(score)
                  
        return clean_scores
            
    def __call__(self, texts, decoding="greedy"):
        scores = self.compute_scores(texts)
        pred_ids = self.decode(scores, decoding=decoding)
        preds = self.verbalize(pred_ids)
        return preds
            
            

