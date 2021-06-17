import pickle
import xgboost
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd

def load_models(m):
    file_name = "model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data[m]
    return model

def format_input(title, author, content):
    corpus =pd.DataFrame([title + " " + author + " " + content]).iloc[0]
    print(corpus)
    stemmer = PorterStemmer()
    #tfidf_vectoriser = TfidfVectorizer(stop_words='english', analyzer=stemmer.stem)
    tfidf_vectoriser = load_models("vectoriser")
    output = tfidf_vectoriser.transform(corpus)
    return output

def predict(x_input):
    model=load_models('model')
    print(type(x_input))

    prediction = model.predict(x_input)
    if prediction:
        prediction = "Fake News"
    else:
        prediction = "Real News"
    pred_0 = model.predict_proba(x_input)[0][0]
    pred_1 = model.predict_proba(x_input)[0][1]
    return [prediction,pred_0,pred_1]


