# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 22:04:54 2021

@author: vikas
"""

f = open("abc.txt", "r")

#Error

f = open("abc.txt", "w")
f.write("Welcom to BA\n")
f.close()

f = open("abc.txt", "a")
f.write("Welcom to analytics\n")
f.write("Welcom to data analytics\n")
f.close()

f = open("abc.txt", "r")
text = f.read()
f.close()
text



f = open('metamorphosis.txt', 'r')
t1 = f.read()
f.close()
t1

t1 = t1.replace('\n',' ')
t1

t1 = t1.replace('  ',' ')
t1

import string
string.punctuation

for i in string.punctuation:
    t1 = t1.replace(i,'')

t1


words = t1.split()
words

'''
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
stem = [ porter.stem(word) for word in words]
stem
'''

import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()


t1 = "This is Good"
t2 = "This is bad"
t3 = "this is best"
t4 = "this is worst"

t5= "Learning is key to success,  it's adapted with continuous being an active learner . Thanks to Henry Harvin they have absolute knowledge on how to guide students or every learner from fresher to experience, they are all providing relevant information in their courses.I am personally doing a Content Writing Course from here and this was my great decision, because across the course there is a lot of relative information although you can gain much confidence in doing assignments. No doubt they have significant experience with all the support which they are providing to you for a lifetime. Personally Henry Harvin is my first choice and also I recommend it to all of you."
for i in string.punctuation:
    t5 = t5.replace(i,'')

t6 = "I have never seen such a fraudulent institute in my life. Please don't go by the name. I did the same mistake. They are only professional till the time you enroll for a course or training program & make payment. After that they don't care about the quality of training being imparted to the learners. I must say there trainers are good for nothing. Extremely poor communication skills and even some trainers don't know basic grammar. Believe me, once you pay the course fee & if you don't like the training program they won't refund you. Forget about refund, they won't pick your calls. You will left frustrated, harassed & cheated by these cheaters."
scores = sid.polarity_scores(t6)
print(scores)






