
import tensorflow as tf
#print(tf.__version__)

import nltk
#nltk.download('punkt')

import pandas as pd
import gensim
from gensim.models import Word2Vec, KeyedVectors

# 2) Data Processing

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

# 4) Predict the Output


