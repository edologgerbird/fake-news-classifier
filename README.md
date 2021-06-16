# Fake News Classifier

## Project Overview

Repository for the fake news classifier project
- Developed models to classify if a given news article is fake or reliable news. Positive class indicates fake news. 
- Constructed the model based on the Kaggle Fake News Dataset which contains 20800 labeled news articles, of which about half are fake news articles and the other half are reliable news articles.
- Inspected for inherent patterns using unsupervised learning (K-means clustering) that will discriminate the news articles between fake and reliable, without referencing to the given labels.
- Conducted both Hashing Vectorisation and IF-IDF Vectorisation of the text data to determine which form of NLP Vectorisation produced to best performing model.
- Tested Passive Aggressive Classifier, Logistic Regression and XGBoost Regression Classifier. 
- XGBoost Regression Classifier on TF-IDF Vectorisaton was the best performing model, with an accuracy of 0.9978 on Test Data.
 
## Codes and Resources Used

**Python Version:** 3.9

**Packages:** pandas, numpy, sklearn, matplotlip, seaborn, nltk

**Web Framework Requirements**: ```pip install -r requirements.txt```

**Data Set:** https://www.kaggle.com/c/fake-news/data

- https://medium.com/@rohithramesh1991/unsupervised-text-clustering-using-natural-language-processing-nlp-1a8bc18b048d

- https://github.com/adriancampos1/Enron_Email_Analysis/blob/master/Enron_Email_Analysis_K-means_clustering.ipynb

- https://www.kaggle.com/haithemhermessi/fake-news-classifier-using-lstm

- https://trenton3983.github.io/files/projects/2019-07-19_fraud_detection_python/2019-07-19_fraud_detection_python.html

- https://github.com/rohithramesh1991/Unsupervised-Text-Clustering

- https://www.kaggle.com/nasirkhalid24/unsupervised-k-means-clustering-fake-news-87

- https://www.kaggle.com/bhavikjain/fake-news-classification 

- https://stackoverflow.com/questions/58120996/text-analysis-finding-the-most-common-word-in-a-column-using-python


 ## Exploratory Data Analysis
 
 - Check for null values
 - Scatter plot of the distribution of article lengths
 - Reported statistical measures of the distribution of article lengths
 - Obtained a list of the top 50 most frequently used word, excluding stopwords

## Data Cleaning and Preparation 

- Remove null text entries
- Combined title, author and article content into 1 column
- Hashing Vectorisation of text
- TF-IDF Vectorisation of text

## Kmeans Clustering (Hashing Vectorisation and IF-IDF Vectorisation)

- First conudct unsupervised learning to check for potential inherent patterns within the data which discriminates fake and real data.
- Using 2 clusters for fake and reliable news
- Building the models
- Report Model Performances

## Passive Aggressive Classifier (Hashing Vectorisation and IF-IDF Vectorisation)

- Building the models
- Report Model Performaces

## Logistic Regression (Hashing Vectorisation and IF-IDF Vectorisation)

- Building the models
- Report Model Performaces

## XGBoost Regression (Hashing Vectorisation and IF-IDF Vectorisation)

- Building the models
- Report Model Performances

## Model Performances

### KMeans Clustering

#### Hashing Vectorisation

Accuracy: 0.4904447

        precision    recall  f1-score   support

    0       0.00      0.00      0.00      8341
    1       0.49      0.98      0.66      8299

#### TF-IDF Vectorisation

Accuracy: 0.4904447

        precision    recall  f1-score   support

    0       0.00      0.00      0.00      8341
    1       0.49      0.98      0.66      8299


Kmeans Clustering performed poorly in clustering the text entries into 2 distinct clusters. There is insufficient patterns inherently present in the text data that could effectively discriminate between real and fake news.

### Passive Aggressive Classifier

#### Hashing Vectorisation

Accuracy: 0.97355769

        precision    recall  f1-score   support

    0       0.96      0.99      0.97      2046
    1       0.99      0.96      0.97      2114


#### TF-IDF Vectorisation

Accuracy: 0.9591346

        precision    recall  f1-score   support

    0       0.92      1.00      0.96      2046
    1       1.00      0.92      0.96      2114

### Logistic Regression

#### Hashing Vectorisation

Accuracy: 0.98125

        precision    recall  f1-score   support

    0       0.97      0.99      0.98      2046
    1       0.99      0.97      0.98      2114

#### TF-IDF Vectorisation

Accuracy: 0.986538

        precision    recall  f1-score   support

    0       0.98      0.99      0.99      2046
    1       0.99      0.98      0.99      2114

### XGBoost Classifier

#### Hashing Vectorisation

Accuracy: 0.997356

        precision    recall  f1-score   support

    0       1.00      1.00      1.00      2046
    1       1.00      1.00      1.00      2114

#### TF-IFD Vectorisation (best performance)

Accuracy: 0.9978365

        precision    recall  f1-score   support

    0       1.00      1.00      1.00      2046
    1       1.00      1.00      1.00      2114


### Best Model: XGBoost on TF-IFD Vectorised Data

## Productionisation