import spacy
nlp = spacy.load("en_core_web_sm")
print(nlp.Defaults.stop_words)

print(len(nlp.Defaults.stop_words))

print(nlp.vocab['always'].is_stop)

nlp.Defaults.stop_words.add('asdf')
print(nlp.vocab['asdf'].is_stop)

nlp.vocab['no'].is_stop = False
print(nlp.Defaults.stop_words)

nlp.Defaults.stop_words.remove['no']

