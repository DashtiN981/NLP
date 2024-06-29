
s1 = "Apple is looking at buying U.K. startup for 1$ billion"
s2 = "San Francisco considers banning sidewalk delivery robots"
s3 = "facebook is hiring a new vice president in U.S."

import spacy 
nlp = spacy.load(name='en_core_web_sm')

doc1 = nlp(s1)
for ent in doc1.ents:
    print(ent.text, ent.label_, str(spacy.explain(ent.label_)))

doc2 = nlp(s2)
for ent in doc2.ents:
    print(ent.text, ent.label_, str(spacy.explain(ent.label_)))

doc3 = nlp(s3)

ORG = doc3.vocab.strings['ORG']

from spacy.tokens import Span
new_ent = Span(doc3, 0, 1, label = ORG)

doc3.ents = list(doc3.ents) + [new_ent]

for ent in doc3.ents:
    print(ent.text, ent.label_, str(spacy.explain(ent.label_)))

from spacy import displacy

displacy.render(docs=doc1, style='ent', options={'ents':['ORG']}, jupyter=True)