#Rule-Based Matching

import spacy

#Import the Matcher library
from spacy.matcher import Matcher
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab) #created matcher object and pass nlp.vocab

# Here matcher is an object that pairs to current Vocab object
# We can add and remove specific named matchers to matcher as needed

# Creating patterns

# create a list, add inside that list add series of dictionaries 
# Hello World can appear i n the following ways,
# 1) Hello World hello world hello WORLD
# 2) Hello-World
pattern_1 = [{'LOWER': 'hello'},{'LOWER': 'world'}]
#pattern_2= [{'LOWER': 'hello'},{'IS_PUNCT': True},{'LOWER': 'world'}]

# 'LOWER', 'IS_PUNCT' are the attributes
# they has to be written in that way only

# Add patterns to matcher object
 
# Add a match rule to matcher, A match rule consists of,
# 1) ID key
# 2) an on_match callback
# 3) one or more patterns 

matcher.add("Hello World",[pattern_1])
# creating a document

doc = nlp(" 'Hello World' are the first two printed words for most of the programmers, printing 'Hello-World' is most common for beginners")

#finding the matches

find_matches = matcher(doc) #passing doc to matcher object and store this in a variable
for match_id, start, end in find_matches:
    string_id = nlp.vocab.strings[match_id]  # Get string representation
    span = doc[start:end]  # get the matched span
    print(match_id, string_id, start, end, span.text)

# it returns output list of tuples
# string ID, index start and inded end

#setting pattern and quantifiers

#Redefine the patterns:
pattern_3= [{'LOWER': 'hello'},{'IS_PUNCT': True, 'OP':'*'},{'LOWER': 'world'}]

matcher.add("Hello World",[pattern_3])

doc_2 = nlp("You can print Hello World or hello world or Hello-World")

find_matches = matcher(doc_2)
print(find_matches)

#********************************************************
#Phrase Matching
import spacy 
nlp = spacy.load('en_core_web_sm')

#Import the Phrasematcher library
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)

phrase_list = ["Barak Obama", "Angela Merkel", "Washington, D.C."]

#Convert each phrase to a document object
phrase_patterns = [nlp(text) for text in phrase_list] # to do that we are using list comprehension

phrase_patterns
#phrase objects are not string

type(phrase_patterns[0])
#they are spacy docs
#thats why we don't have any quotes there

#pass each doc object into the matcher
matcher.add("TerminologyList", None, *phrase_patterns)
#thats we have to add asterisk mark before phrase_pattern

doc_3 = nlp("German Chancellor Angela Merkel and US President Barak Obama "
           " converse in the Oval office inside the White House in Washington, D.C.")

find_matches = matcher(doc_3) # passin doc to matcher object and store this in variable
print(find_matches)

# define a function to find the matches

for match_id, start, end in find_matches:
    string_id = nlp.vocab.strings[match_id] # get string representation
    span = doc_3[start:end] # get the matched span
    print(match_id, string_id, start,end, span.text)

