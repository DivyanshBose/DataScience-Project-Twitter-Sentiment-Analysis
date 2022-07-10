# About The Dataset

The objective of this task is to detect hate speech in tweets. For the sake of simplicity, we say a tweet contains hate speech if it has negative sentiment associated with it. So, the task is to classify negative statements from other tweets.
Formally, given a training sample of tweets and labels, where label '1' denotes the tweet is negative and label '0' denotes the tweet is positive, your objective is to predict the labels on the test dataset.
For training the models, we provide a labelled dataset of 31,962 tweets. The dataset is provided in the form of a csv file with each line storing a tweet id, its label and the tweet.

# Import necessary Modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import re
import string
import nltk
import warnings
%matplotlib inline
warnings.filterwarnings('ignore')

# Loading the Dataset

df = pd.read_csv('C:\\Users\\Divyansh Bose\\OneDrive\\Documents\\DataScience IV Sem Year 21-22\\DATASET\\twitter.csv')
df.head()

#datatype info
df.info()

# Preprocessing the Dataset

#removes pattern in input text
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i,"",input_txt)
    return input_txt

#remove twitter handles(@user)
df['clean_tweets'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")

#remove special characters, numbers and punctuations
df['clean_tweets'] = df['clean_tweets'].str.replace("[^a-zA-Z#]"," ")
df.head()

#remove short words
df['clean_tweets'] = df['clean_tweets'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3])) 
df.head()

#individual words considered as tokens
tokenized_tweet = df['clean_tweets'].apply(lambda x:x.split())
tokenized_tweet.head()

#stem the words
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])
tokenized_tweet.head()

#combine words into single sentence
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = " ".join(tokenized_tweet[i])
df['clean_tweets'] = tokenized_tweet
df.head()

# Exploratory Data Analysis

#visualize the frequent words
all_words = " ".join([sentence for sentence in df['clean_tweets']])

#extract the hashtag
def hashtag_extract(tweet):
    hashtags = []
    
    #loop words in the tweet
    for word in tweet:
        ht = re.findall(r"#(\w+)",word)
        hashtags.append(ht)
    return hashtags

#extract hashtags from positive tweets
ht_positive = hashtag_extract(df['clean_tweets'][df['label']==0])

#extract hashtags from negative tweets
ht_negative = hashtag_extract(df['clean_tweets'][df['label']==1])

#unnest list
ht_positive = sum(ht_positive,[])
ht_negative = sum(ht_negative,[])

freq = nltk.FreqDist(ht_positive)
d = pd.DataFrame({'Hashtag':list(freq.keys()),'Count':list(freq.values())})
d.head()

#select top 10 hashtags
d = d.nlargest(columns = 'Count',n=10)
plt.figure(figsize=(15,9))
sns.barplot(data=d,  x= 'Hashtag', y = 'Count')
plt.show()

freq = nltk.FreqDist(ht_negative)
d = pd.DataFrame({'Hashtag':list(freq.keys()),'Count':list(freq.values())})
d.head()

#select top 10 hashtags
d = d.nlargest(columns = 'Count',n=10)
plt.figure(figsize=(15,9))
sns.barplot(data=d,  x= 'Hashtag', y = 'Count')
plt.show()

# Coorelation Matrix

#feature extraction
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df = 0.90, min_df = 2, max_features = 1000, stop_words = 'english')
bow = bow_vectorizer.fit_transform(df['clean_tweets'])

# Test Train Split

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(bow,df['label'], random_state=42, test_size=0.25)

# Model Training

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#training
model = LinearRegression()
model.fit(x_train,y_train)

#Testing
y_pred = model.predict(x_test)
y_pred

#mean squared error
mse = mean_squared_error(y_test,y_pred)
mse

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score,accuracy_score,mean_squared_error

#training
model = GaussianNB()
model.fit(x_train.toarray(),y_train)

#Testing
pred = model.predict(x_test.toarray())
f1_score(y_test,pred)

#accuracy
acc = accuracy_score(y_test,pred)*100
acc

#mean squared error
mse = mean_squared_error(y_test,y_pred)
mse

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score,mean_squared_error

#training
model = LogisticRegression()
model.fit(x_train,y_train)

#Testing
pred = model.predict(x_test)
f1_score(y_test,pred)

#accuracy
acc = accuracy_score(y_test,pred)*100
acc

#mean squared error
mse = mean_squared_error(y_test,y_pred)
mse

