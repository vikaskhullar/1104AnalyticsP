# -*- coding: utf-8 -*-

import tweepy  #pip install tweepy
import csv
import pandas as pd

#credentials
'''
https://dev.twitter.com/apps/new
'''

APIKey='swVgSVaFyTQy5Mfermv9Ailtk'
APISecret='ZlABfZXGrkz5TnCh1ppQc26Q5fDKH7COBDGSnht0uhREaZRVJD'
AccessToken='144501392-EsDZpYReefhUXJWF0u0FKPg5tp1DtZECPICYGvgG'
AccessTokenSecret='PwZaF8CTXxaolK3PivNNoGitjqrXR8ARRcr5nDbvqbOoa'



auth = tweepy.OAuthHandler(APIKey, APISecret)
auth.set_access_token(AccessToken, AccessTokenSecret)
api = tweepy.API(auth, wait_on_rate_limit=True)



handle =['Reliance Industries']

#change to see the values, tweets will be stored in du.csv in project folder #run the for loop together

#geo='28.644800,77.216721,1000km'
#, geocode=geo
'''
csvFile = open('DSC.csv','w')
csvWriter = csv.writer(csvFile)
'''


import pandas as pd

creat = []
txt = []

for tweets in api.search_tweets(q=handle, count =100, lang="en"):
    print(tweets.created_at, tweets.text.encode('utf-8'))
    creat.append(tweets.created_at)
    txt.append(tweets.text)
    
txt
creat


df = pd.DataFrame({'creat':creat, 'txt':txt})
df.to_csv("tweet.csv")


import string
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()


scr = []
for i in range(0, len(df)):
    tx = df.iloc[i]['txt']
    for i in string.punctuation:
        tx = tx.replace(i,'')
    scores = sid.polarity_scores(tx)
    scr.append(scores)
    

df['score'] = scr

df.to_csv('tweetscore.csv')
