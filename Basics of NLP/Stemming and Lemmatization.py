#Stemming
words = ['run','runner','running','ran','runs','easily','fairly']

import nltk

from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

p_stemmer = PorterStemmer()
s_stemmer = SnowballStemmer(language='english')

for word in words:
    print(word + '-----'+p_stemmer.stem(word))
    print(word + '-----'+s_stemmer.stem(word))


#Lemmatization
import spacy
nlp = spacy.load(name="en_core_web_md")
doc1 = nlp("The striped bats are hanging on their feet for best")
for token in doc1:
    print(token.text, '\t', token.lemma_)

s1 = "The striped bats are hanging on their feet for best"
for word in s1.split():
    print(word + '-----'+p_stemmer.stem(word))