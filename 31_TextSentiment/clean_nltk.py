from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# load data
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

# split into sentences
#sentences = sent_tokenize(text)
#print(sentences[0])

# split into words
tokens = word_tokenize(text)

#stop_words = stopwords.words('english')
#print(stop_words)

## remove all tokens that are not alphanumeric
#words = [word for word in tokens if word.isalpha()]
#print(words[:100])

# stemming of words
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in tokens]
print(stemmed[:100])


import re
doc = "NLP  is an interesting     field.  "
new_doc = re.sub("s+"," ", doc)
print(new_doc)

"I like NLP." == 'I like NLP'


text = "Hello! How are you!! I'm very excited that you're going for a trip to Europe!! Yayy!"
re.sub("[^-9A-Za-z ]", "" , text)




 

#Removing extra spaces

import re
doc = "NLP  is an interesting     field.  "
new_doc = re.sub("s+"," ", doc)
print(new_doc)

 

#Removing punctuations

"I like NLP." == 'I like NLP'

#Punctuations can be removed by using regular expressions.

CODE:

text = "Hello! How are you!! I'm very excited that you're going for a trip to Europe!! Yayy!"
re.sub("[^-9A-Za-z ]", "" , text)

#Punctuations can also be removed by using a package from the string library.


import string
text = "Hello! How are you!! I'm very excited that you're going for a trip to Europe!! Yayy!"
text_clean = "".join([i for i in text if i not in string.punctuation])
text_clean



#Case Normalization

import string
text = "Hello! How are you!! I'm very excited that you're going for a trip to Europe!! Yayy!"
text_clean = "".join([i.lower() for i in text if i not in string.punctuation])
text_clean

#Tokenization

text = "Hello! How are you!! I'm very excited that you're going for a trip to Europe!! Yayy!"
nltk.tokenize.word_tokenize(text)



text = "Hello! How are you!! I'm very excited that you're going for a trip to Europe!! Yayy!"
from nltk.tokenize import TweetTokenizer
tweet = TweetTokenizer()
tweet.tokenize(text)


import re
a = 'What are your views related to US elections @nitin'
re.split('s@', a)

Removing Stopwords

stopwords = nltk.corpus.stopwords.words('english')
text = "Hello! How are you!! I'm very excited that you're going for a trip to Europe!! Yayy!"
text_new = "".join([i for i in text if i not in string.punctuation])
print(text_new)
words = nltk.tokenize.word_tokenize(text_new)
print(words)
words_new = [i for i in words if i not in stopwords]
print(words_new)

'''
Stemming: A technique that takes the word to its root form. It just removes suffixes from the words. The stemmed word might not be part of the dictionary, i.e it will not necessarily give meaning. There are two main types of stemmer- Porter Stemmer and Snow Ball Stemmer(advanced version of Porter Stemmer).
'''

ps = nltk.PorterStemmer()
w = [ps.stem(word) for word in words_new]
print(w)

ss = nltk.SnowballStemmer(language = 'english')
w = [ss.stem(word) for word in words_new]
print(w)

'''
Lemmatization: Takes the word to its root form called Lemma. It helps to bring words to their dictionary form. It is applied to nouns by default. It is more accurate as it uses more informed analysis to create groups of words with similar meanings based on the context, so it is complex and takes more time. This is used where we need to retain the contextual information.
'''
wn = nltk.WordNetLemmatizer()
w = [wn.lemmatize(word) for word in words_new]
print(w)

