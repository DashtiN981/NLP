# 1) ********Importing the Data set ***************

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_yelp = pd.read_csv("D:/MyProject/Naghme/NLP/IMDB, Amazoo and Yelp review Classification/yelp.txt", sep='\t', header=None) 

#print(data_yelp.head())

# review and sentiment
# 0-Negative, 1-Positive for positive review

# Assign column names
column_name = ['Review','Sentiment']
data_yelp.columns = column_name

print(data_yelp.head()) 
print(data_yelp.shape)

data_amazon = pd.read_csv("D:/MyProject/Naghme/NLP/IMDB, Amazoo and Yelp review Classification/amazon.txt", sep='\t', header=None)
data_amazon.columns = column_name

print(data_amazon.head())
print(data_amazon.shape)

data_imdb = pd.read_csv("D:/MyProject/Naghme/NLP/IMDB, Amazoo and Yelp review Classification/imdb.txt", sep='\t', header=None)
data_imdb.columns = column_name

print(data_imdb.head())
print(data_imdb.shape)

# Append all the data in a single dataframe

data = data_yelp.append([data_amazon, data_imdb], ignore_index = True)
print(data.shape)

# 1386 Positive reviews
# 1362 Negative reviews
print(data['Sentiment'].value_counts())
print(data.isnull().sum())

x = data['Review']
y = data['Sentiment']

# 2) ********* Data Cleaning **********************

# here we will remove stopwords, punctiations
# as well as we will apply lemmatization



# Create a function to clean the data

import string
punct = string.punctuation

from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS) # list of stopwords

# creating a function for data cleaning
import spacy

nlp = spacy.load('en_core_web_sm')

def text_data_cleaning(sentence):
    doc = nlp(sentence)

    tokens = [] # list of tokens
    for token in doc:
        if token.pos_ == "PRON":
            continue
        temp = token.lemma_.lower().strip()
        tokens.append(temp)

    cleaned_tokens = []
    for token in tokens:
        if token not in punct and token not in stopwords:
          cleaned_tokens.append(token)   
    return cleaned_tokens

print(text_data_cleaning("Hello all, It's a beautiful day outside there!"))

# Vectorization Feature Engineering(TF-IDF)
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

tfidf = TfidfVectorizer(tokenizer=text_data_cleaning)
# tokenizer = text_data_cleaning, tokenization will be done according to this function

classifier = LinearSVC()
# 3) ********* Train the Model ********************


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

#print(x_train.shape,x_test.shape)
#2198 samples in training dataset and 550 in test dataset

#print(x_train.head())

# Fit the x_train and y_train

clf = Pipeline([("tfidf",tfidf),("clf",classifier)])
#it will first do vectorization and thenn it will do classification

clf.fit(x_train, y_train)
# in this we don't neet to prepare the dataset for testing(x_test)

# 4) *********** Predict the Test set Result **************
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = clf.predict(x_test)

#confusion_matrix
print(confusion_matrix(y_test, y_pred))

#classification_report
print(classification_report(y_test,y_pred))
# we are getting almost 77% accuracy

print(accuracy_score(y_test, y_pred))
# 76% accuracy

print(clf.predict(["Wow, I am learning Natural Language Processing in fun fashion!"]))
#output is 1. that means review is positive

print(clf.predict(["It's hard to learn new things!"]))
#output is 0, that means review is Negative