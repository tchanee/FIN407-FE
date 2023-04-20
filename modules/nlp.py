from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from typing import List

#################################################################
# Named Entity Recognition Task
#################################################################

class BertWrapperForNER():
    DEFAULT_MODEL = "Davlan/distilbert-base-multilingual-cased-ner-hrl"
    
    def __init__(self, model_name=DEFAULT_MODEL):
        self.model_name = model_name.split("/")[-1]
        
        self.org_start_label = "B-ORG"
        self.org_middle_label = "I-ORG"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForTokenClassification.from_pretrained(model_name)
         
        self.pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
    
    def eval(self, texts: List):
        return self.pipeline(texts)
        
    def recognize(self, texts: List):
        outputs = self.eval(texts)
        
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
            
    def save_to(self, path):
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)
        
    @staticmethod
    def load_from(path):
        return BertWrapperForNER(model_name=path)
    

#################################################################
# Sentiment Analysis Task
#################################################################

class BERTWrapperForSA():
    def __init__(self, model_name=None):
        if model_name is not None:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForSequenceClassification.from_pretrained(model_name)
        
    def __call__(self, tweets: List):
        return self.pipeline(tweets)
    
    def load_from_local(self, path):
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model = BertForSequenceClassification.from_pretrained(path)
            
    def save_to_local(self, path):
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)

class VaderSentimentWrapper():
    pass

