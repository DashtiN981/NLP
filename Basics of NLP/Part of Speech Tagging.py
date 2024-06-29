s1 = "Apple is looking at bying U.K. startup for $1 billion"

import spacy
nlp = spacy.load(name ='en_core_web_sm')

doc = nlp(s1)

for token in doc:
    print(token.text, token.pos_, token.tag_, spacy.explain(token.tag_))

for key, val in doc.count_by(spacy.attrs.POS).items():
    print(key, doc.vocab[key].text, val)

from spacy import displacy

displacy.render(docs=doc , style= 'dep', jupyter=True)
