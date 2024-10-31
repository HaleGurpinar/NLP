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
from wordcloud import WordCloud
from PIL import Image
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

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