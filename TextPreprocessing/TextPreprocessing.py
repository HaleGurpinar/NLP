# ###########################################################
# Introduction to Text Mining and Natural Language Processing
# ###########################################################

# Sentiment Analysis and Sentiment Modeling for Amazon Reviews

# !pip install nltk
# !pip install textblob
# !pip install wordcloud


from warnings import filterwarnings
import pandas as pd
from nltk.corpus import stopwords

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
