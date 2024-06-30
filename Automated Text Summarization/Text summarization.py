# Text summarization
text = "Rafael Nadal, born on June 3, 1986, in Manacor, Mallorca, Spain, is widely regarded as one of the greatest tennis players of all time. His relentless pursuit of excellence on the court, combined with his humble upbringing and fierce competitiveness, has made him a legend in the sport. Nadal's journey to stardom began at a young age, honing his skills on the rocky terrain of Mallorca, which helped develop his unique playing style characterized by exceptional footwork, agility, and a powerful topspin forehand. He turned professional in 2001 and quickly rose through the ranks, capturing his first ATP title just two years later. Nadal's dominance on clay courts is legendary, with a record 13 French Open titles, the most in the Open Era. His victory at Roland Garros in 2008, where he became the first player to win the tournament without dropping a set, stands as a testament to his unparalleled skill and mental toughness. Off clay, Nadal has also found success on other surfaces, including hard courts and grass, winning major titles at the US Open and Wimbledon. His ability to adapt his game to various conditions and opponents has been a hallmark of his career. Beyond his remarkable achievements on the court, Nadal's impact extends off it. He has been an active participant in charitable causes, particularly those focused on children's health and education. His Rafael Nadal Foundation works towards improving educational opportunities for underprivileged children, reflecting his commitment to giving back to society. Nadal's legacy in tennis is not just about his numerous titles and records; it's about his relentless spirit, his dedication to the sport, and his influence beyond the tennis world. As Nadal continues to compete at the highest level, his legacy grows with every match he plays. His story serves as an inspiration to aspiring athletes around the globe, reminding them that with passion, hard work, and resilience, anything is possible. Whether he's chasing down shots on the baseline or advocating for social change, Rafael Nadal remains a beacon of excellence and integrity in the world of sports"

# 1) ********** Importing the libraries and Dataset*********
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

nlp = spacy.load("en_core_web_sm")

doc = nlp(text)

tokens = [token.text for token in doc]
print(tokens)

punctuation = punctuation + '\n'

# 2)*************** Text Cleaning ***********************

# first we will try to create word frequency counter
word_freq = {}

stop_words= list(STOP_WORDS)

for word in doc:
    if word.text.lower() not in stop_words:
        if word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text] = 1
            else:
                word_freq[word.text] += 1
print(word_freq)

max_freq = max(word_freq.values())

for word in word_freq.keys():
    word_freq[word] = word_freq[word]/max_freq

print(word_freq)

# 3) ************** Sentence tokenization ***************

sent_tokens = [sent for sent in doc.sents]
print(sent_tokens)

# provide a score for every single sentence

sent_score ={}

for sent in sent_tokens:
    for word in sent:
        if word.text.lower() in word_freq.keys():
            if sent not in sent_score.keys():
                sent_score[sent] = word_freq[word.text.lower()]
            else :
                sent_score[sent] += word_freq[word.text.lower()]
print(sent_score)

# 4) ************ Select 30% sentences with maximum score ******************

from heapq import nlargest

print(len(sent_score))

# 5)************ Getting Summary *****************

summary = nlargest(n = 5, iterable= sent_score , key= sent_score.get)

print(summary)

final_summary = [word.text for word in summary]
print(final_summary)

summary = " ".join(final_summary)

print(summary)

print(len(summary)/len(text))
