# FIN407-FE
Repository for FIN-407 Financial Econometrics project.

## Tasks

Abiola: ETL (main task)
Johnny: summary stats (main task) + keyword filtering (side task ?) + ETL (side task)
Farouk: FinBert, NER-DistilBERT, VaderSentiment (main task)
Safae: market definition (main task) + stock and crypto data (main task) + summary stats with Johnny (side task)


## Approaches & Tools

Approach 1: 
- clustering the tweets by industies/companies: keyword filtering or Davlan/distilbert-base-multilingual-cased-ner-hrl
- sentiment analysis: vaderSentiment

Approach 2:
- clustering the tweets by industies/companies: Davlan/distilbert-base-multilingual-cased-ner-hrl
- sentiment analysis: FinBert

## Report Outline

0) abstract (+ motivation)
    a) field intro (NLP applied to market prediction, maybe cite one or two previous public works)
    b) motivation
1) Introduction
    a) the goal of the project
    b) a brief overview of the best approach (just a few words)
    c) the most important result
2) ELT (Extract Load Transform)
  i) data acquisition
    a) source
    b) format
  ii)  data cleaning
    a) what processing was done on the data?
    b) what assumptions?
  iii) data exploration
    a) data description through some basic summary statistics: vocabulary size, number of tweets available, average sentiment, concentration of tweets over time, tweet length, tweet views
    comment: plots before cleaning or after cleaning or both ?
4) Analysis
    i) description of the approach
        a) what market?
        b) describe the approach
  i) Approach 1
     a) limitations/tradeoffs
     b) implementation and test
  ii) Approach 2
     a) limitations/tradeoffs
     b) implementation and test
  iii) Approach 3
     a) limitations/tradeoffs
     b) implementation and test
5) Conclusion
  - summary of our main results
