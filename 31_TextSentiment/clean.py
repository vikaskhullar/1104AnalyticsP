import string
import re

# load text

filename = 'metamorphosis.txt'

file = open(filename, 'rt')

text = file.read()
text
file.close()


text = text.replace('\n',' ')
text
# split into words by white space
words = text.split()
words

# prepare regex for char filtering
re_punc = re.compile('[%s]' % re.escape(string.punctuation))

re_punc
# remove punctuation from each word

stripped = [re_punc.sub('', w) for w in words]

print(stripped)



from nltk.stem.porter import PorterStemmer
# stemming of words
porter = PorterStemmer()

stemmed = [porter.stem(word) for word in stripped]

print(stemmed[:100])



#from nltk.stem import WordNetLemmatizer  as wnl
  
#lemmatizer =wnl() 
  
#lemeted = [lemmatizer.lemmatize(word) for word in stripped]

#print(lemeted[:100])



import nltk

nltk.download('vader_lexicon')
nltk.download('punkt')

# first, we import the relevant modules from the NLTK library

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# next, we initialize VADER so we can use it within our Python script
sid = SentimentIntensityAnalyzer()


import re
words_re = re.compile(" ".join(stemmed))
word_str = str(words_re)
word_str

scores = sid.polarity_scores(word_str)


scores

test1 = "I am good"
test2 = "This is bad"
test3 = "Bad worst Bad worst  Bad worst  Bad worst  Bad worst  Bad worst  Do not value students at all, provide the same course at different prices. Make false promises, which are not followed up.Would not recommend."



scores = sid.polarity_scores(test3)
scores

test4 = "One of the best platforms to get trained in content writing by  expert faculties. It’s great for beginners to launch their career in writing, learning the technical aspects. All the queries and concerns are covered by the trainer. In depth insight into all trending topics makes the course more relevant. Commendable service by allotting relationship managers for students to solve their technical issues. CDCW course by Henry Harvin is best for aspiring writers to build their passion into profession. Doing the course I witnessed a drastic development in my writing style and started to notice mistakes in my writing and correct them respectively. It helped me in reviving my writing skills by giving a refine touch."

# prepare regex for char filtering
re_punc = re.compile('[%s]' % re.escape(string.punctuation))

re_punc
# remove punctuation from each word

words = test4.split()
words
stripped = [re_punc.sub('', w) for w in words]

print(stripped)


import re
words_re = re.compile(" ".join(words))
word_str = str(words_re)
word_str

scores = sid.polarity_scores(word_str)
scores







