# How to install Stanford Core NLP for use in Python - https://stackabuse.com/python-for-nlp-getting-started-with-the-stanfordcorenlp-library/
# Navigate inside the unzip folder twice as it creates on additional directory inside

#!pip install pycorenlp
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

'''
# You can parse the data this way as well

import requests
url = 'http://localhost:9000'
request_params = {'outputFormat': 'json'}
text = "This is a test sentence."
r = requests.post(url,data=text,params=request_params)
print (r.json())
'''


doc = "I like this chocolate. This chocolate is not good. The chocolate is delicious. Its a very tasty chocolate. This is so bad"
doc = "I can not stress enough how much I love the chocolate"
doc = "This is extremely awesome"
doc = "This is very bad"
doc = "This is extremely awesome. This is bad"
doc = "This is the worst. This is pathetic"

annot_doc = nlp.annotate(doc,
    properties={
       'annotators': 'sentiment',
       'outputFormat': 'json',
       'timeout': 1000,
    })

sentiment_score = 0
for sentence in annot_doc["sentences"]:
    print ( " ".join([word["word"] for word in sentence["tokens"]]) + " => " \
        + str(sentence["sentimentValue"]) + " = "+ sentence["sentiment"])

    if int(sentence["sentimentValue"])==1:
        sentiment_score = -1 + sentiment_score
    else:
        sentiment_score = int(sentence["sentimentValue"]) + sentiment_score

    print(sentiment_score)


