# ###########################################################
# ##############  TEXT PREPROCESSING  #######################
# ###########################################################

# ###########################################################
# Introduction to Text Mining and Natural Language Processing
# ###########################################################

# Sentiment Analysis and Sentiment Modeling for Amazon Reviews

# !pip install nltk
# !pip install textblob
# !pip install wordcloud


from warnings import filterwarnings

import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from PIL import Image
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn import preprocessing

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# ###########################################################
# 1. Text Preprocessing
df = pd.read_csv("amazon_reviews.csv", sep=",")
print(df.head(5))

df['reviewText'] = df['reviewText'].str.lower()  # Normalization(normalize case folding)

# ###########################################################
# 2. Punctuations(get rid of  punctuations)
df['reviewText'] = df['reviewText'].str.replace(r'[^\w\s]', '', regex=True)
print(df)

# ###########################################################
# 3. Numbers(get rid of numbers )
df['reviewText'] = df['reviewText'].str.replace(r'\d', '', regex=True)
print(df)

# ###########################################################
# 4. Stop Words(get rid of stop words(common words as "is, that, are, of, ..." ))
import nltk
nltk.download('stopwords')
sw = stopwords.words('english')
print(sw)

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
print(df)

# ###########################################################
# 5. Rare Words(get rid of rare words)
temp_df = pd.Series(' '.join(df['reviewText']).split()).value_counts()  # All words frequency
print(temp_df)

drops = temp_df[temp_df <= 1]  # Used just once words
print(drops)

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))
print(df)

# ###########################################################
# 6. Tokenization(dividing sentences in their parts)
import nltk
from textblob import TextBlob

nltk.download("punkt")
print(df['reviewText'].apply(lambda x: TextBlob(x).words).head())

# ###########################################################
# 7. Lemmatization(reduction words to original form (books -> book))  #stemming(generally not using)
import nltk
from textblob import Word
nltk.download("wordnet")

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
print(df['reviewText'])  # capabilities -> capability, things -> thing...


# ###########################################################
# ###########  TEXT VISUALIZATION  ##########################
# ###########################################################

# ###########################################################
# 1. Calculating Term Frequency
tf = df["reviewText"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
print(tf)

print(tf.sort_values("tf", ascending=False))

# ###########################################################
# 2. Bar Plot
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

# ###########################################################
# 3. Word Cloud
text = " ".join(i for i in df.reviewText)  # Agg all rows to a text
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("wordcloud.png")  # you can save image to your project file as wanted format

# ###########################################################
# 4. Word Cloud by Templates
tr_mask = np.array(Image.open("TR_flag.jpg"))

wordcloud = WordCloud(max_words=1000,
                      background_color="white",
                      mask=tr_mask,
                      contour_width=3,
                      contour_color="firebrick")
wordcloud.generate(text)
plt.figure(figsize=[10, 10])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# ###########################################################
# ###########  SENTIMENT MODELLING  #########################
# ###########################################################


# ###########################################################
# 1. Sentiment Analysis *****************************
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

print(sia.polarity_scores("The film was awesome"))  # Key is compound. If compound >0 positive sentence else negative.
print(sia.polarity_scores("I like this music but it is not good as the other one"))  # negative sentence example

df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x))  # scores for all rows but dictionary structure
print(df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"]))

df["polarity_score"] = df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])  # permanent modifying

# ###########################################################Important###################################################
# 1. Future Engineering
df["reviewText"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"] = df["reviewText"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
print(df["sentiment_label"])
print(df["sentiment_label"].value_counts())
print(df.groupby("sentiment_label")["overall"].mean())
df["sentiment_label"] = preprocessing.LabelEncoder().fit_transform(df["sentiment_label"])

y = df["sentiment_label"]
X = df["reviewText"]

# ##############################################################################################################
# Count Vectors

# Count Vectors: frequency representations
# TF-IDF Vectors: normalized frequency representations
# Word Embeddings (Word2Vec, GloVe, BERT vs.)

# words - numeric repr. of words

# characters - numeric repr. of characters

# ngram
a = """I'm gonna show this instance in a longer text to be able to understand. 
N-grams represent combinations of words used together and are used to generate features."""
print(TextBlob(a).ngrams(5))

# Count Vector
from  sklearn.feature_extraction.text import CountVectorizer

# Aim: Vectorized text - every rows

corpus = ['This is the first document.',
          'This document is the second document.',
          'And this is the third one.',
          'Is this the first document?']

# word frequency
vectorizer =CountVectorizer()
X_c = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out()) # unique names(words) in corpus  -- columns for array
print(X_c.toarray())

# n-gram frequency
vectorizer2 =CountVectorizer(analyzer='word', ngram_range=(2, 2))
X_n = vectorizer2.fit_transform(corpus)
print(vectorizer2.get_feature_names_out())
print(X_n.toarray())

vectorizer =CountVectorizer()
X_count = vectorizer.fit_transform(X)
print(vectorizer.get_feature_names_out()[10:15]) # unique names(words) in corpus  -- columns for array
print(X_count.toarray()[10:15])