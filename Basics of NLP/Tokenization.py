import spacy
nlp = spacy.load(name="en_core_web_md")
s1 = "welome to NLP training"
doc1 = nlp(s1)
for token in doc1:
    print(token)
print(len(doc1))