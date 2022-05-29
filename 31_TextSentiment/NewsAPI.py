# importing requests package
import requests	

def NewsFromBBC():
    query_params = {"source": "bbc-news","sortBy": "top","apiKey": "4dbc17e007ab436fb66416009dfb59a8"	}
    main_url = " https://newsapi.org/v1/articles"
    res = requests.get(main_url, params=query_params)
    open_bbc_page = res.json()
    article = open_bbc_page["articles"]
    results = []
    for ar in article:
        results.append(ar["title"])	
    return(results)

result = NewsFromBBC()

result


    
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

   

# next, we initialize VADER so we can use it within our Python script
sid = SentimentIntensityAnalyzer()



for res in result:
    '''
    words = res.split()
    stripped = [re_punc.sub('', w) for w in words]
    words_re = re.compile(" ".join(stripped))    
    word_str = str(words_re)
    #print(str(word_str))
    '''
    scores = sid.polarity_scores(res)
    print (res)
    print (scores)

