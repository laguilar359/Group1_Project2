# Group1_Project2

# I. Dependencies
 ### *Web Scraping Code*
 
>#!pip install bs4

>#!pip install selenium

>from selenium import webdriver

>from selenium.webdriver.common.by import By

>from selenium.webdriver.support.ui import 

>WebDriverWait

>from selenium.webdriver.support import 

>expected_conditions as EC

>from selenium.webdriver.common.action_chains import ActionChains

>from bs4 import BeautifulSoup

>import requests

>import pandas as pd

>import numpy as np

>from pathlib import Path

### *AAPL_WEB_SCRAPING_CODE*
>from splinter import Browser

>import pandas as pd

>import numpy as np

### *NLP-VADER*
##### Tokens
>from dotenv import load_dotenv

>import nltk

>nltk.download('vader_lexicon')

>from nltk.sentiment.vader import SentimentIntensityAnalyzer

>from vaderSentiment import SentimentIntensityAnalyzer

>from pathlib import Path

>analyzer = SentimentIntensityAnalyzer()

##### Most Frequent Tokens
>from collections import Counter

##### WordClouds
>from wordcloud import WordCloud

>import matplotlib.pyplot as plt

>plt.style.use('seaborn-whitegrid')

>import matplotlib as mpl

>mpl.rcParams['figure.figsize'] = [20.0, 10.0]

##### NER
>import spacy

>from spacy import displacy

# II. Assumptions

# III. Summary of Findings

# IV. Questions Answered

# V. What we would change if we had the time