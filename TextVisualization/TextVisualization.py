# ###########################################################
# Text Visualization
# ###########################################################

from warnings import filterwarnings
import pandas as pd
import nltk
from textblob import Word
nltk.download("wordnet")
import nltk
from textblob import TextBlob

nltk.download("punkt")
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv("amazon_reviews.csv", sep=",")

# Calculating Term Frequency
tf = df["reviewText"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
print(tf.columns)