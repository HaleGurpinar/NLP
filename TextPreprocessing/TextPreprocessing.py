# ###########################################################
# Introduction to Text Mining and Natural Language Processing
# ###########################################################

# Sentiment Analysis and Sentiment Modeling for Amazon Reviews

# !pip install nltk
# !pip install textblob
# !pip install wordcloud


from warnings import filterwarnings
import pandas as pd

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# ###########################################################
# 1. Text Preprocessing

df = pd.read_csv("amazon_reviews.csv", sep=",")
df.head(5)
