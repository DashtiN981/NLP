
# 1) ************** Data Processing ******************
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('D:/MyProject/Naghme/NLP/Spam Message Classification/spam.tsv', sep='\t')
print(df.head())

#print(df.isna().sum())

#print(df.tail())

#print(df['label'].value_counts())

ham = df[df['label'] == 'ham']
spam = df[df['label'] == 'spam']

#print(ham.shape, spam.shape)

ham = ham.sample(spam.shape[0])
data = ham.append(spam,ignore_index = True)

#print(data.shape)
#print(data['label'].value_counts())

#print(data.head())

plt.hist(data[data['label'] == 'ham']['length'], bins= 100, alpha = 0.7)
plt.hist(data[data['label'] == 'spam']['length'], bins= 100, alpha = 0.7)
plt.show()

plt.hist(data[data['label'] == 'ham']['punct'], bins= 100, alpha = 0.7)
plt.hist(data[data['label'] == 'spam']['punct'], bins= 100, alpha = 0.7)
plt.show()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.3, random_state=0, shuffle= True)
print(X_train.shape,X_test.shape)

# 2) ************** Building the Model(Random Forest) ******************

from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

classifier = Pipeline([("tfidf", TfidfVectorizer()),("classifier", RandomForestClassifier( n_estimators=100))])
classifier.fit(X_train, y_train)

# 3) ************** Prediction the results(Random Forest) ******************

y_pred = classifier.predict(X_test)
#print(y_test, y_pred)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 4) ************** Building the Model(SVM) ******************

from sklearn.svm import SVC

svm = Pipeline([("tfidf", TfidfVectorizer()),("classifier", SVC( C = 100, gamma='auto' ))])
svm.fit(X_train,y_train)

# 5) ************** Prediction the results(SVM) ******************

y_pred = svm.predict(X_test)
#print(y_test, y_pred)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6) ************** Prediction for created data ******************

test1 = ["Hello, you are learning natrual language processing"]
test2 = ["Hope you are doing good and learning new things"]
test3 = ["Congratulations, you won a lottory ticket worth 1$ Million ! to claim call on 446677"]

print(classifier.predict(test1))
print(classifier.predict(test2))
print(classifier.predict(test3))

print(svm.predict(test1))
print(svm.predict(test2))
print(svm.predict(test3))
