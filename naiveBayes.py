'''
Nikki Agrawal
Wes Chao
Intro to Machine Learning
Naive Bayes Data Analysis

Resources
https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html

'''

from sklearn.datasets import fetch_20newsgroups # the 20 newgroups set is included in scikit-learn
from sklearn.naive_bayes import MultinomialNB # we need this for our Naive Bayes model

# These next two are about processing the data. We'll look into this more later in the semester.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# which newsgroups we want to download --> I added a lot more newsgroups so it would be more accurate
newsgroup_names = [
 'alt.atheism',
 'comp.graphics',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space'
 ]

 #all possible news news newgroups
'''
'alt.atheism',
'comp.graphics',
'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware',
'comp.sys.mac.hardware',
'comp.windows.x',
'misc.forsale',
'rec.autos',
'rec.motorcycles',
'rec.sport.baseball',
'rec.sport.hockey',
'sci.crypt',
'sci.electronics',
'sci.med',
'sci.space',
'soc.religion.christian',
'talk.politics.guns',
'talk.politics.mideast',
'talk.politics.misc',
'talk.religion.misc'

'''

# get the newsgroup data (organized much like the iris data)
newsgroups = fetch_20newsgroups(categories=newsgroup_names, shuffle=True, random_state=265)
newsgroups.keys()

# Convert the text into numbers that represent each word (bag of words method)
word_vector = CountVectorizer()
word_vector_counts = word_vector.fit_transform(newsgroups.data)

# Account for the length of the documents:
#   get the frequency with which the word occurs instead of the raw number of times
term_freq_transformer = TfidfTransformer()
term_freq = term_freq_transformer.fit_transform(word_vector_counts)

# Train the Naive Bayes model
model = MultinomialNB().fit(term_freq, newsgroups.target)

# Predict some new fake documents
fake_docs = [
    'That GPU has amazing performance with a lot of shaders',
    'The player had a wicked slap shot',
    'I spent all day yesterday soldering banks of capacitors',
    'Today I have to solder a bank of capacitors',
    'NASA has rovers on Mars',
    'Hello World, it is I',
    'Sometimes, Math is really boring',
    'I am a Hindu',
    "Ew, broccoli",
    "Take me out to the Ball Game, Take me out to the Crowd",
    "A Tiger is a Cat",
    "How do you spell supercalifragilisticexpialidocious?",
    "The CIA has rovers in Israel"
    ]
fake_counts = word_vector.transform(fake_docs)
fake_term_freq = term_freq_transformer.transform(fake_counts)

predicted = model.predict(fake_term_freq)
probabilities = model.predict_proba(fake_term_freq)

print('Predictions:')

for doc, group in zip(fake_docs, predicted):
    print('\t{0} => {1}'.format(doc, newsgroups.target_names[group]))

print('Probabilities:')
print(''.join(['{:20}'.format(name) for name in newsgroups.target_names]))

for probs in probabilities:
    print(''.join(['{:17.8}'.format(prob) for prob in probs]))

#played around to try and fix the formatting
'''
predicted = model.predict(fake_term_freq)
print('Predictions:')
for doc, group in zip(fake_docs, predicted):
    print('\t{0} => {1}'.format(doc, newsgroups.target_names[group]))

probabilities = model.predict_proba(fake_term_freq)
print('Probabilities:')
for probs in probabilities:
    answer = '{:5}'.format(name) for name in newsgroups.target_names + '{:<5}'.format(prob) for prob in probs
    print(''.join(answer))
'''

#original code
'''
predicted = model.predict(fake_term_freq)
probabilities = model.predict_proba(fake_term_freq)

print('Predictions:')

for doc, group in zip(fake_docs, predicted):
    print('\t{0} => {1}'.format(doc, newsgroups.target_names[group]))

print('Probabilities:')
print(''.join(['{:17}'.format(name) for name in newsgroups.target_names]))

for probs in probabilities:
    print(''.join(['{:<17.8}'.format(prob) for prob in probs]))
'''
