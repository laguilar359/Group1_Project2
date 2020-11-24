#Initial Imports
import os
import pandas as pd


from dotenv import load_dotenv
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path
analyzer = SentimentIntensityAnalyzer()


load_dotenv()
get_ipython().run_line_magic("matplotlib", " inline")


#Import Data from .CSV files
csv_path = aapl_file = Path('../Resources/AAPL_HEADLINES.csv')
# csv_path = (r'C:\Users\annmi\OneDrive\Desktop\Class\Project\Group1_Project2\Web-Scraping-APP\AAPL_HEADLINES.csv')


aapl_headlines = pd.read_csv(csv_path)
aapl_headlines.head()


#Initial Imports
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from string import punctuation
import re


# Expand the default stopwords list if necessary
sw_addons={'char', 'colleague', 'day', 'another', 'edition', 'writes', 'whats','going', 'new', 'could', 'say'}


lemmatizer=WordNetLemmatizer()

def tokenizer(text):
    '''Remove Characters'''
    regex = re.compile("[^a-zA-Z ]")
    text = regex.sub('', text)
    '''Create a list of the words'''
    words=word_tokenize(text)
    '''Remove the stop words'''
    sw = set(stopwords.words('english')).union(set(punctuation)).union(sw_addons)
    '''Lemmatize Words into root words'''
    lem = [lemmatizer.lemmatize(word) for word in words]
    '''Convert the words to lowercase'''
    tokens = [word.lower() for word in lem if word.lower() not in sw]
    return tokens


def add_tokens(df):
    '''Apply tokenizer to Headlines'''
    df['Tokens']=df['Headline'].apply(tokenizer)
    '''Reset Index'''
    df=df.reset_index()
    '''Reset Index to Date'''
    df=df.set_index('Date', inplace=True)
    return df



add_tokens(aapl_headlines)
aapl_headlines.head()


#Initial Imports
from collections import Counter


#Create the Tokenizer Function
def token_count(df, N=20):
    """Returns the top N tokens from the frequency count"""
    '''Creates a Big String'''
    big_string=df["Headline"].str.cat()
    '''Runs Tokenizer on Big String'''
    df_tokenized=tokenizer(big_string)
    '''Counts Tokens'''
    return Counter(df_tokenized).most_common(N)


token_count(aapl_headlines)


#Function if there are different column titles
def token_count(text_column, N=20):
    """Returns the top N tokens from the frequency count"""
    '''Creates a Big String'''
    big_string=text_column.str.cat()
    '''Runs Tokenizer on Big String'''
    df_tokenized=tokenizer(big_string)
    '''Counts Tokens'''
    return Counter(df_tokenized).most_common(N)

token_count(aapl_headlines['Headline'])


from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [20.0, 10.0]


#Create WC function
def word_cloud(df):
    '''Create Big String'''
    big_string=df['Headline'].str.cat()
    '''Tokenize Big String'''
    df_tokenized=tokenizer(big_string)
    '''Create Second Big String of Tokenized Words'''
    wc_string=' '.join(df_tokenized)
    '''Set the Word Cloud Object'''
    nlp_wc = WordCloud(width=1200, height=800, max_words=50).generate(wc_string)
    return plt.imshow(nlp_wc)


word_cloud(aapl_headlines)


# WC function for different column titles
def word_cloud(text_column):
    '''Create Big String'''
    big_string=text_column.str.cat()
    '''Tokenize Big String'''
    df_tokenized=tokenizer(big_string)
    '''Create Second Big String of Tokenized Words'''
    wc_string=' '.join(df_tokenized)
    '''Set the WordCloud Object'''
    nlp_wc = WordCloud(width=1200, height=800, max_words=50).generate(wc_string)
    return plt.imshow(nlp_wc)


word_cloud(aapl_headlines['Headline'])


#Imports
import spacy
from spacy import displacy


# Create NER function
def named_entity_recognition(df):
    '''Load the spaCy model'''
    nlp = spacy.load('en_core_web_sm')
    '''Create a big string'''
    big_string=df['Headline'].str.cat()
    '''Run Processor'''
    processed_doc=nlp(big_string)
    '''Add Title'''
    processed_doc.user_data['title']='Named Entity Recognition'
    '''Render the visualization'''
    return displacy.render(processed_doc, style='ent')


#Create NER function for column titles
def named_entity_recognition(text_column):
    '''Load the spaCy model'''
    nlp = spacy.load('en_core_web_sm')
    '''Create a big string'''
    big_string=text_column.str.cat()
    '''Run Processor'''
    processed_doc=nlp(big_string)
    '''Add Title'''
    processed_doc.user_data['title']='Named Entity Recognition'
    '''Render the visualization'''
    return displacy.render(processed_doc, style='ent')


named_entity_recognition(aapl_headlines['Headline'])



