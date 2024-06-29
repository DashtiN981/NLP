
#s1 = "This is a sentence. This is second sentence. This is last sentence"

#import spacy
#nlp = spacy.load(name='en_core_web_sm')
#doc1 = nlp(s1)
#for sent in doc1.sents:
 #   print(sent.text)

import spacy
from spacy.language import Language

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Define a custom component to set sentence boundaries
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == ";":
            doc[token.i + 1].is_sent_start = True
    return doc

# Register the custom component as a factory
Language.component("set_custom_boundaries", func=set_custom_boundaries)

# Add the custom component using its string name
nlp.add_pipe("set_custom_boundaries", before="parser")

# Your input sentence
s2 = "This is a sentence; This is the second sentence; This is the last sentence"

# Process the text
doc_2 = nlp(s2)

# Print the modified sentences
for sent in doc_2.sents:
    print(sent.text)