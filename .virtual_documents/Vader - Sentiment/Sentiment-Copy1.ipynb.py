# Initial imports
import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import numpy as np

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

get_ipython().run_line_magic("matplotlib", "inline")



nltk.download("vader_lexicon")
analyzer = SentimentIntensityAnalyzer()



# Load .env enviroment variables
load_dotenv()


# Set Alpaca API key and secret
alpaca_api_key = os.getenv('ALPACA_API_KEY')
alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')

api = tradeapi.REST(alpaca_api_key, alpaca_secret_key, api_version='v2')


def stock_info_grab(ticker):
    """
    Takes ticker symbol and returns DataFrame with Date, Close, and Pct Change columns.
    """
    # Set timeframe to '1D'
    timeframe = "1D"

    # Set current date and the date from one month ago using the ISO format
    current_date = pd.Timestamp("2020-11-09", tz="America/New_York").isoformat()
    past_date = pd.Timestamp("2016-08-27", tz="America/New_York").isoformat()

    df = api.get_barset(
        ticker,
        timeframe,
        limit=None,
        start=past_date,
        end=current_date,
        after=None,
        until=None,
    ).df
    df = df.droplevel(axis=1, level=0)
    df.index = df.index.date
    df['pct change'] = df['close'].pct_change()
    df['pct change'].dropna
    df = df.reset_index()
    df = df.drop(columns=['open', 'high', 'low', 'volume'])
    df = df.rename(columns={'index':'Date'})
    df = df.set_index('Date')
    return df


aapl_stock_info = stock_info_grab("AAPL")
amzn_stock_info = stock_info_grab("AMZN")
tsla_stock_info = stock_info_grab("TSLA")
spy_stock_info = stock_info_grab("SPY")
amzn_stock_info


aapl_file = Path('Resources/AAPL_HEADLINES.csv')
amzn_file = Path('Resources/AMZN_HEADLINES.csv')
spy_file = Path('Resources/SPY_HEADLINES.csv')
tsla_file = Path('Resources/TSLA_HEADLINES.csv')

aapl_headlines_df = pd.read_csv(aapl_file)
amzn_headlines_df = pd.read_csv(amzn_file)
spy_headlines_df = pd.read_csv(spy_file)
tsla_headlines_df = pd.read_csv(tsla_file)

#aapl_headlines['Date'] = pd.to_datetime(aapl_headlines['Date']).dt.strftime('get_ipython().run_line_magic("Y-%m-%d')", "")
#aapl_headlines = aapl_headlines.set_index('Date')
amzn_headlines_df


def get_sentiment(score):
    """
    Calculates the sentiment based on the compound score.
    """
    result = 0  # Neutral by default
    if score >= 0.05:  # Positive
        result = 1
    elif score <= -0.05:  # Negative
        result = -1

    return result


def create_sentiment_df(df):
    """
    Takes headlines DataFrame & creates DataFrame with Sentiment columns.
    Splits Date & Time, creates Time column and moves Date to Index.
    """
    title_sent = {
        "compound": [],
        "positive": [],
        "neutral": [],
        "negative": [],
        "sentiment": [],
    }

    for index, row in df.iterrows():
        try:
            # Sentiment scoring with VADER
            title_sentiment = analyzer.polarity_scores(row["Headline"])
            title_sent["compound"].append(title_sentiment["compound"])
            title_sent["positive"].append(title_sentiment["pos"])
            title_sent["neutral"].append(title_sentiment["neu"])
            title_sent["negative"].append(title_sentiment["neg"])
            title_sent["sentiment"].append(get_sentiment(title_sentiment["compound"]))
        except AttributeError:
            pass

    title_sent_df = pd.DataFrame(title_sent)
    #title_sent_df.head()

    headline_sentiment_df = df.join(title_sent_df)
    headline_sentiment_df.dropna()
    headline_sentiment_df['Date'] = headline_sentiment_df['Date'].str.replace('at','-')
    headline_sentiment_df['Date'] = headline_sentiment_df['Date'].str.split('-').str[0]
    headline_sentiment_df = headline_sentiment_df.reindex(columns=['Date', 'Headline', 'compound', 'positive', 'neutral', 'negative', 'sentiment'])
    headline_sentiment_df['Date'] = pd.to_datetime(headline_sentiment_df['Date'])
    headline_sentiment_df.set_index('Date')
    return headline_sentiment_df


#issue with amzn_headlines --- need to fix
aapl_headlines = create_sentiment_df(aapl_headlines_df)
#amzn_headlines = create_sentiment_df(amzn_headlines_df)
tsla_headlines = create_sentiment_df(tsla_headlines_df)
spy_headlines = create_sentiment_df(spy_headlines_df)



# find average sentiment score by date

aapl_scores = aapl_headlines.groupby('Date').mean().sort_values(by='Date')
#amzn_scores = amzn_headlines.groupby(['Date']).mean().sort_values(by='Date')
tsla_scores = tsla_headlines.groupby(['Date']).mean().sort_values(by='Date')
spy_scores = spy_headlines.groupby(['Date']).mean().sort_values(by='Date')
aapl_scores



# TO DO: drop compund col on all scores
aapl_scores = aapl_scores.drop(columns='compound')
#amzn_scores = amzn_scores.drop(columns='compound')
tsla_scores = tsla_scores.drop(columns='compound')
spy_scores = spy_scores.drop(columns='compound')


tsla_scores


# sent scores distribution across each df poss use histogram, calc meanstd, or percentiles 
aapl_complete = pd.concat([aapl_scores,aapl_stock_info], join='outer', axis=1).dropna()
#amzn_complete = pd.concat([amzn_scores,amzn_stock_info], join='outer', axis=1).dropna()
tsla_complete = pd.concat([tsla_scores,tsla_stock_info], join='outer', axis=1).dropna()
spy_complete = pd.concat([spy_scores,spy_stock_info], join='outer', axis=1).dropna()
aapl_complete





# TO DO: shift aapl_complete['pct change'] one day on all dfs
# TO DO: dropna() on all df['predicted pct change'] cols 
aapl_complete['predicted pct change'] = aapl_complete['pct change'].shift()
#amzn_complete['predicted pct change'] = amzn_complete['pct change'].shift()
tsla_complete['predicted pct change'] = tsla_complete['pct change'].shift()
spy_complete['predicted pct change'] = spy_complete['pct change'].shift()
aapl_complete


aapl_complete = aapl_complete.dropna()
#amzn_complete = amzn_complete.dropna()
tsla_complete = tsla_complete.dropna()
spy_complete = spy_complete.dropna()



aapl_complete


def get_sentiment(df):
    """
    Calculates the sentiment based on the compound score.
    """
    result = [
        (df.iloc[:,3] >= 0.01),
        (df.iloc[:,3] <= 0.00)
    ]
    
    values = [0, 1]
    
    df['buy/sell'] = np.select(result, values)
    
    return df


aapl_complete_sentiment = get_sentiment(aapl_complete)
#amzn_complete_sentiment = get_sentiment(amzn_complete)
tsla_complete_sentiment = get_sentiment(tsla_complete)
spy_complete_sentiment = get_sentiment(spy_complete)
aapl_complete_sentiment


def regression_analysis(df):
    y = df['buy/sell']
    X = df.drop(columns=['buy/sell', 'pct change', 'close', 'positive', 'neutral', 'negative', 'sentiment'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1,  stratify=y)

    classifier = LogisticRegression(solver='lbfgs', random_state=1)
    classifier.fit(X_train, y_train)
    print(f"Training Data Score: {classifier.score(X_train, y_train)}")
    print(f"Testing Data Score: {classifier.score(X_test, y_test)}")
    predictions = classifier.predict(X_test)
    results = pd.DataFrame({"Prediction": predictions, "Actual": y_test}).reset_index(drop=True)
    return results


regression_analysis(tsla_complete_sentiment)


y = aapl_complete_sentiment['buy/sell']
X = aapl_complete_sentiment.drop(columns=['buy/sell', 'pct change', 'close', 'positive', 'neutral', 'negative', 'sentiment'])


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1,  stratify=y)

X_train.shape


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs', random_state=1)
classifier


classifier.fit(X_train, y_train)


print(f"Training Data Score: {classifier.score(X_train, y_train)}")
print(f"Testing Data Score: {classifier.score(X_test, y_test)}")


predictions = classifier.predict(X_test)
results = pd.DataFrame({"Prediction": predictions, "Actual": y_test}).reset_index(drop=True)
results.head(20)






