# FIN407-FE
Repository for FIN-407 Financial Econometrics project.

## Approach: do elon musk tweets move the market?
TODO

## Tasks

week 9:
Abiola:
    - extract the tweets that elon musk replied to in a (separate) file, and augment the current processed dataframe with the original tweet's ID for each reply
    - approach?
Johnny:
    - sentiment analysis exploration: average sentiment across all (replies and tweets), replies separateley and tweets separately
    - word cloud approach to market definition: which companies/stocks/organizations has elon musk tweeted the most about?
    - approach?
Safae:
    - create a pipeline to extract WRDS data for elon musk companies + crypto currencies
    - approach?
everyone: report parts

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
