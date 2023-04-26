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
    
    def extract(self, texts):
        return self.pipeline(texts)
        
    def __call__(self, texts):
        outputs = self.extract(texts)
        
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
    DISTILBERT_ID = "textattack/distilbert-base-cased-SST-2"
    
    def __init__(self, model_id_or_path: str, verbalizer=None):
        self.model_name = model_id_or_path.split("/")[-1]
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id_or_path)
        self.model.eval()
        
        self.verbalizer = verbalizer
    
    def verbalize(self, pred_ids):
        if self.verbalizer is None:
            # default verbalizer
            return [self.model.config.id2label[pred_id.item()] for pred_id in pred_ids.flatten()]
        # custom verbalizer
        return [self.verbalizer(pred_id.item()) for pred_id in pred_ids.flatten()]
        
    def decode(self, probas, decoding, rebalancing_threshold=0.3):
        if decoding == "argmax":
            pred_ids = probas.argmax(dim=-1)
            return pred_ids 
        elif decoding == "rebalanced":
            rebalancing_threshold = 0.3
            pos_id = self.model.config.label2id["positive"]
            neg_id = self.model.config.label2id["negative"]
            neu_id = self.model.config.label2id["neutral"]
            mask = (probas[:, pos_id] + probas[:, neg_id] >= rebalancing_threshold).flatten()
            probas[mask, neu_id] = 0
            pred_ids = probas.argmax(dim=-1)
            return pred_ids
        else:
            raise ValueError(f"Decoding method '{decoding}' not supported")
         
    def predict(self, texts):
        with torch.no_grad():
            batch_encoding = self.tokenizer.batch_encode_plus(texts, return_tensors="pt", truncation=True, padding=True)
            logits = self.model(**batch_encoding).logits
            probas = torch.softmax(logits, dim=-1)
            return probas
    
    def __call__(self, texts, decoding="argmax"):
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
    LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
    ID2LABEL = {v:k for k, v in LABEL2ID.items()}
    
    def __init__(self, verbalizer=None):
        self.model = SentimentIntensityAnalyzer()
        self.score_func = np.frompyfunc(self.model.polarity_scores, 1, 1)
        
        self.verbalizer = verbalizer
        
    def verbalize(self, pred_ids):
        if self.verbalizer is None:
            return [VaderSentimentWrapper.ID2LABEL[pred_id] for pred_id in pred_ids]
        return [self.verbalizer(pred_id) for pred_id in pred_ids]
       
    def decode(self, scores, decoding="argmax", rebalancing_threshold=0.3):
        scores = np.array([list(score.values())[:-1] for score in scores])
        if decoding == "argmax":
            pred_ids = np.argmax(scores, axis=1).tolist()
            return pred_ids
        elif decoding == "rebalanced":
            pos_id = VaderSentimentWrapper.LABEL2ID["positive"]
            neu_id = VaderSentimentWrapper.LABEL2ID["neutral"]
            neg_id = VaderSentimentWrapper.LABEL2ID["negative"]
            mask = (scores[:, pos_id] + scores[:, neg_id] >= rebalancing_threshold).flatten()
            scores[mask, neu_id] = 0
            pred_ids = np.argmax(scores, axis=1).flatten().tolist()
            return pred_ids
    
    def compute_scores(self, texts):
        scores = self.score_func(np.array(texts)).tolist()
        return scores
            
    def __call__(self, texts, decoding="argmax"):
        scores = self.compute_scores(texts)
        pred_ids = self.decode(scores, decoding=decoding)
        preds = self.verbalize(pred_ids)
        return preds
            
            

