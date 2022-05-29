# -*- coding: utf-8 -*-
"""
Created on Wed May  4 21:07:12 2022

@author: vikas
"""

# Read, Write, Append

f = open("abc.txt", 'w')

f.write("Welcome To Python\n")
f.write("Analytics\n")
f.write("Databases")

f.close()

f = open('abc.txt', 'r')
txt=f.readline()
while(txt!=''):
    print(txt)
    txt=f.readline()
f.close()    

f = open('abc.txt', 'r')
f.read(10)
f.close()    


f = open("abc.txt", 'a')
f.write("\nBusiness Analytics")
f.close()



# List to csv data

rno=''
name=''
sem =''
sub =''

f = open('dat111.csv', 'w')


def stud():
    rno=input("Enter R No->")
    name= input("Enter Name->")
    sem = input("Enter Sem->")
    sub = input("Enter Subject->")
    return(rno, name, sem, sub)


f.write('rno' +','+ 'name' +','+ 'sem' +','+ 'sub\n')

for i in range(5):    
    rno, name,sem, sub = stud()
    f.write(rno +','+ name +','+ sem +','+ sub+'\n')

f.close()



# Text Cleaning

import string
filename = 'metamorphosis.txt'
f = open(filename)
text = f.read()
text

pun = string.punctuation

for i in pun:
    text1=text.replace(i, '')
text1


import re
text2 = re.sub("[^-9A-Za-z ]", "" , text)

text2

doc = "NLP  is an interesting     field.  "
text3 =  re.sub(" +"," ", text2 )

text3

words = text3.split(" ")
stripped = [w.replace(' ','') for w in words]


from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

stemmed = [porter.stem(word) for word in stripped]


import nltk

nltk.download('vader_lexicon')
nltk.download('punkt')


from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

test1 = "This is good"
test2 =" This is Best"
test3 = "This is bad"
test4 = "This is Worst"
test5 = "This is normal"


sid.polarity_scores(test5)


test6 = "Henry Harvin truly has changed my life!!! It is an amazing institute for learning the Spanish language, having an experienced and encouraging faculty. The teachers make everything seem easy and fun and there is always a friendly atmosphere in the class. When I started learning Spanish, I had never thought I would come so far, this could become possible under the guidance of Pankaj Sir and other teachers. I truly believe that at Henry Harvin anyone can learn anything!!!! Plus with lots of fun. The best institution for languages.."
sid.polarity_scores(test6)


test7 = "Not a good choice for anyone who is serious about his/her studies and career! I got enrolled in the Creative Writing weekend batch in October 2021. After 2-3 sessions, the mentor started canceling the classes every other weekend without any prior intimation. Even after contacting their support via their site and telegram group multiple times, we received no response. Later we got to know they have changed the class timings on their own. Missed a few classes due to this misunderstanding and one day out of the blue I received a call from their representative that the classes are already done and there is one 'bonus' class for us that we can join to resolve our 'queries'. There are no videos available on the site for the said batch on the site. I requested customer support via chat to provide the recordings but instead of asking questions about my batch, they shared with me a link to a site page where, even after spending a good amount of time, I am not able to find a single recording of the classes. Without any guidance from their team, I don't think I would be able to complete this course now and request for the certification. I have wasted a good amount of money and time on this and I suggest you all to properly inquire about everything related to the mentor, timing, assignments, and course duration before paying your hard-earned money to this institute. The contact person will not answer your calls or whatsapp messages once you get enrolled. DONT pay anything without getting an assurance that the institute's support team will be available for you for your course-related concerns whenever needed."
sid.polarity_scores(test7)



#News Sentiment Reviewer



from newsapi import NewsApiClient

# Init
newsapi = NewsApiClient(api_key='ec4604b2a608472dbf57f26b938ed8b9')

# /v2/top-headlines
top_headlines = newsapi.get_top_headlines(q='NSE',
                                          language='en',
                                          country='us')


print(top_headlines)
type(top_headlines)

top_headlines.keys()

type(top_headlines['articles'])
len(top_headlines['articles'])
top_headlines['articles'][0]
type(top_headlines['articles'][0])
top_headlines['articles'][0].keys()

text = top_headlines['articles'][0]['description']


from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
sid.polarity_scores(text)


from newsapi import NewsApiClient

# Init
newsapi = NewsApiClient(api_key='ec4604b2a608472dbf57f26b938ed8b9')

# /v2/top-headlines
top_headlines = newsapi.get_everything(q='NSE',
                                          language='en')


print(top_headlines)
type(top_headlines)

top_headlines.keys()

type(top_headlines['articles'])
top_headlines['articles'][0].keys()

top_headlines['totalResults']

text=[]
for i in range(0,len(top_headlines['articles'])):    
    text.append(top_headlines['articles'][i]['description'])


from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
pol=[]
for txt in text:
    pol.append(sid.polarity_scores(txt)['compound'])


import pandas as pd

pd.DataFrame({"Text":text, "Polarity":pol}).to_csv("NewsSentiment.csv")


#Tweet Analytics











import tweepy  #pip install tweepy
import csv
import pandas as pd

APIKey='14eiLam7Zlf7j7obr6PJT0GM0'
APISecret='TEBbIeucTiWKjMZI43fsfmO0J53Ihkbl2X2tPDCW7HvUFGBeDo'
AccessToken='144501392-oDIM3EXINkV6QjjrLDqhs7T9wN81rc34CP9rdvnk'
AccessTokenSecret='xpyS7VKoYaFaWSwE334oTCFDhq4gHwUc3Rn2BuGL5016C'


auth = tweepy.OAuthHandler(APIKey, APISecret)
auth.set_access_token(AccessToken, AccessTokenSecret)
api = tweepy.API(auth, wait_on_rate_limit=True)

handle = ['NSE']


crat=[]
txt=[]
for tweets in api.search_tweets(q=handle, count =100, lang="en"):
    print(tweets.created_at, tweets.text.encode('utf-8'))
    crat.append(tweets.created_at)
    txt.append(tweets.text)
    
txt


df = pd.DataFrame({"CreateAt":crat, "TweetText":txt})



from nltk.tokenize import TweetTokenizer
tweet = TweetTokenizer()
tweet.tokenize(txt[0])


def cleaner(tweet):
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = " ".join(tweet.split())
    tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) \
         if w.lower() in words or not w.isalpha())
    return tweet

df['TweetText'] = df['TweetText'].map(lambda x: cleaner(x))
df.to_csv('tweet.csv') #specify location



























































