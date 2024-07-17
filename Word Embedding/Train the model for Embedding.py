
import tensorflow as tf
#print(tf.__version__)

import nltk
#nltk.download('punkt')

import pandas as pd
import gensim
from gensim.models import Word2Vec, KeyedVectors

# 2) Data Preprocessing

df = pd.read_csv('D:/MyProject/Naghme/NLP/Word Embedding/reddit_worldnews_start_to_2016-11-22.csv')

print(df.head())

news_titles = df['title'].values
#print(news_titles)

# tokenize the words
new_vec = [nltk.word_tokenize(title) for title in news_titles]

print(new_vec[0])

# 3) Build the model

model = Word2Vec(new_vec, min_count=1, vector_size=32)
# text, min word count, size of each vector

#print(model)

# 4) Predict the Output

# find 10 closet words in the vector space that we have created
print(model.wv.most_similar('man'))

# see the vector

#print(model.wv['man'])
vector_of_man = model.wv.get_vector('man')

print(vector_of_man)

# so this is how man is represented in out vector space

#let us try the famous relationship
vec = model.wv['king'] - model.wv['man'] + model.wv['women']
print(model.wv.most_similar([vec]))


vec = model.wv['Germany'] - model.wv['Berlin'] + model.wv['Paris']
print(model.wv.most_similar([vec]))

