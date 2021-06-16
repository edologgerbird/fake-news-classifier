# Fake News Classifier

## Project Overview

Repository for the fake news classifier project
- Developed models to classify if a given news article is fake or reliable news. Positive class indicates fake news. 
- Constructed the model based on the Kaggle Fake News Dataset which contains 20800 labeled news articles, of which about half are fake news articles and the other half are reliable news articles.
- Inspected for inherent patterns using unsupervised learning (K-means clustering) that will discriminate the news articles between fake and reliable, without referencing to the given labels.
- Conducted both Hashing Vectorisation and IF-IDF Vectorisation of the text data to determine which form of NLP Vectorisation produced to best performing model.
- Tested Passive Aggressive Classifier, Logistic Regression and XGBoost Regression Classifier. 
- XGBoost Regression Classifier on TF-IDF Vectorisaton was the best performing model, with an accuracy of 0.9978 on Test Data.
 
 ## Steps:
 1. EDA

 2. Data cleaning

    - Remove null text entries

3.  Data Preparation

    - Combining text columns
    - Hashing Vectorisation
    - TF-IDF Vectorisation

4. Kmeans Clustering (Hashing Vectorisation and IF-IDF Vectorisation)

    - First conudct unsupervised learning to check for potential inherent patterns within the data which discriminates fake and real data.
    - Using 2 clusters for fake and reliable news
    - Building the models
    - Model Performances

5. Passive Aggressive Classifier (Hashing Vectorisation and IF-IDF Vectorisation)

    - Building the models
    - Model Performaces

6. Logistic Regression (Hashing Vectorisation and IF-IDF Vectorisation)

    - Building the models
    - Model Performaces

7. XGBoost Regression (Hashing Vectorisation and IF-IDF Vectorisation)

    - Building the models
    - Model Performances

8. Best Model for Fake News Classification


## Resources:
- https://medium.com/@rohithramesh1991/unsupervised-text-clustering-using-natural-language-processing-nlp-1a8bc18b048d

- https://github.com/adriancampos1/Enron_Email_Analysis/blob/master/Enron_Email_Analysis_K-means_clustering.ipynb

- https://www.kaggle.com/haithemhermessi/fake-news-classifier-using-lstm

- https://www.kaggle.com/c/fake-news/data

- https://trenton3983.github.io/files/projects/2019-07-19_fraud_detection_python/2019-07-19_fraud_detection_python.html

- https://github.com/rohithramesh1991/Unsupervised-Text-Clustering

- https://www.kaggle.com/nasirkhalid24/unsupervised-k-means-clustering-fake-news-87

- https://www.kaggle.com/lykin22/fake-news-classification-nlp?select=train.csv

- https://www.kaggle.com/bhavikjain/fake-news-classification 

