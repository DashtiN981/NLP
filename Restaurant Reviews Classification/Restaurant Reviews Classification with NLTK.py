# 1) ************** Business Model *****************************************

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("D:/MyProject/Naghme/NLP/Restaurant Reviews Classification/RestaurantReviews.tsv", sep='\t', quoting=3 )
print(data.head())

print(data['Liked'].value_counts())

# 2) ************** Cleaning Text data *************************************

import nltk
import re       # this is regular experession library

nltk.download('stopwords') # import stopwords

from nltk.corpus import stopwords
print(data['Review'][0])

review = re.sub('[^a-zA-Z]', ' ', data['Review'][0])  # keep only alphabets and digits
print(review)

review = review.lower()   # transform to lowercase
review = review.split()   # split the review like tokenization

print(review)
stopwords.words('english')

preview = []
for word in review:
    if word not  in stopwords.words('english'):
        preview.append(word)    # remove stopwords

print(preview)

review = [word for word in review if word not  in stopwords.words('english')]
print(review)

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

review = [ps.stem(word) for word in review]      #apply stemming process
print(review)

review = " ".join(review)
print(review)

#doing the previous steps for all reviews in data by creating corpus
corpus = []

ps = PorterStemmer()

for i in range(len(data)):

    review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])
    review = review = review.lower()
    review = review = review.split()
    review = [ps.stem(word) for word in review if word not  in stopwords.words('english')]
    review = " ".join(review)
    
    corpus.append(review)

print(corpus)

# 3) *************** Bag of Word Model *************************************

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)

x = cv.fit_transform(corpus).toarray() # create a spars matrix of words in each review
                                        # total number of rows in matrix will be defined by the totals number of records(reviews)
                                        # total number of columns in matrix is 1500
print(x.shape)

y = data.iloc[:, 1].values

print(y.shape)

print(y[:10])

# 4) **************** Apply Naive bayes Algorithm ****************************

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=0)

print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred))








