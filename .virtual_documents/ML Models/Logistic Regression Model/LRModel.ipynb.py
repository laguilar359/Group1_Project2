# Initial imports
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import tensorflow as tf
get_ipython().run_line_magic("matplotlib", "inline")
get_ipython().run_line_magic("matplotlib", " inline")


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
aapl_stock_info


aapl_file = Path('../Resources/AAPL_HEADLINES.csv')
amzn_file = Path('../Resources/AMZN_HEADLINES.csv')
spy_file = Path('../Resources/SPY_HEADLINES.csv')
tsla_file = Path('../Resources/TSLA_HEADLINES.csv')

aapl_headlines_df = pd.read_csv(aapl_file)
amzn_headlines_df = pd.read_csv(amzn_file)
spy_headlines_df = pd.read_csv(spy_file)
tsla_headlines_df = pd.read_csv(tsla_file)

#aapl_headlines['Date'] = pd.to_datetime(aapl_headlines['Date']).dt.strftime('get_ipython().run_line_magic("Y-%m-%d')", "")
#aapl_headlines = aapl_headlines.set_index('Date')
aapl_headlines_df


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


aapl_headlines = create_sentiment_df(aapl_headlines_df)
#amzn_headlines = create_sentiment_df(amzn_headlines_df)
tsla_headlines = create_sentiment_df(tsla_headlines_df)
spy_headlines = create_sentiment_df(spy_headlines_df)
aapl_headlines





# find average sentiment score by date
aapl_scores = aapl_headlines.groupby('Date').mean().sort_values(by='Date')
#amzn_scores = amzn_headlines.groupby(['Date']).mean().sort_values(by='Date')
tsla_scores = tsla_headlines.groupby(['Date']).mean().sort_values(by='Date')
spy_scores = spy_headlines.groupby(['Date']).mean().sort_values(by='Date')


aapl_scores.head()


# TO DO: drop compund col on all scores
aapl_scores = aapl_scores.drop(columns='compound')
#amzn_scores = amzn_scores.drop(columns='compound')
tsla_scores = tsla_scores.drop(columns='compound')
spy_scores = spy_scores.drop(columns='compound')


# sent scores distribution across each df poss use histogram, calc meanstd, or percentiles 
aapl_complete = pd.concat([aapl_scores,aapl_stock_info], join='outer', axis=1).dropna()
#amzn_complete = pd.concat([amzn_scores,amzn_stock_info], join='outer', axis=1).dropna()
tsla_complete = pd.concat([tsla_scores,tsla_stock_info], join='outer', axis=1).dropna()
spy_complete = pd.concat([spy_scores,spy_stock_info], join='outer', axis=1).dropna()
aapl_complete


# TO DO: shift aapl_complete['pct change'] one day on all dfs
# TO DO: dropna() on all df['predicted pct change'] cols 
aapl_complete['predicted pct change'] = aapl_complete['pct change'].shift(periods=-1)
#amzn_complete['predicted pct change'] = amzn_complete['pct change'].shift(periods=-1)
tsla_complete['predicted pct change'] = tsla_complete['pct change'].shift(periods=-1)
spy_complete['predicted pct change'] = spy_complete['pct change'].shift(periods=-1)



aapl_complete = aapl_complete.dropna()
#amzn_complete = amzn_complete.dropna()
tsla_complete = tsla_complete.dropna()
spy_complete = spy_complete.dropna()



def get_sentiment(df):
    """
    Calculates the sentiment based on the compound score.
    """
    result = [
        (df['predicted pct change'] >= 0.00),
        (df['predicted pct change'] < 0.00)
    ]
    
    values = [1, 0]
    
    df['target'] = np.select(result, values)
    
    return df


aapl_complete_sentiment = get_sentiment(aapl_complete)
#amzn_complete_sentiment = get_sentiment(amzn_complete)
tsla_complete_sentiment = get_sentiment(tsla_complete)
spy_complete_sentiment = get_sentiment(spy_complete)
aapl_complete_sentiment


df = aapl_complete_sentiment


df



X = df.copy()
X = df[["positive", "neutral", "negative", "sentiment"]].values
#X = X.reshape(-1, 1)
X[:5]


X.shape



y = df["target"].values
#y = df["pct change"].shift(periods=1).values
y = y.reshape(-1, 1)
y[:5]


y.shape


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, 
                                                   y, 
                                                   random_state=1, 
                                                   stratify=y)
X_train.shape


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs', random_state=1)
classifier


classifier.fit(X_train, y_train.ravel())


print(type(y_train))


print(f"Training Data Score: {classifier.score(X_train, y_train)}")
print(f"Testing Data Score: {classifier.score(X_test, y_test)}")


predictions = classifier.predict(X_test)



from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)


from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))





























# Define features data
#X = df[['positive', 'neutral', 'negative','sentiment']].values
#X = X.drop(columns=["pct change"])


#X[:5]

X = df.copy()
X =df['sentiment'].values
#X = df[['positive', 'neutral', 'negative','sentiment']].values
#X = X.drop(columns=["close", "pct change", "predicted pct change"]).values
X = X.reshape(-1, 1)
X[:5]


# Define target data
y = df["pct change"].values
#y = df["pct change"].shift(periods=1).values
y = y.reshape(-1, 1)
y[:5]


# Create training and testing datasets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)


# Create the scaler instance
from sklearn.preprocessing import StandardScaler

X_scaler = StandardScaler()


# Fit the scaler
X_scaler.fit(X_train)


# Scale the features data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# Define the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

number_inputs = 1
number_hidden_nodes = 4

nn = Sequential()
nn.add(Dense(units=number_hidden_nodes, input_dim=number_inputs, activation="relu"))
nn.add(Dense(1, activation="sigmoid"))


# Compile model
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# Fit the model
model = nn.fit(X_train_scaled, y_train, epochs=100)


# Create a dataframe with the history dictionary
df_plot = pd.DataFrame(model.history, index=range(1, len(model.history["loss"]) + 1))

# Plot the loss
df_plot.plot(y="loss")


# Plot the accuracy
df_plot.plot(y="accuracy")


# Evaluate the model fit with linear dummy data
model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")





# Define the model with "Hard Sigmoid" activation
number_inputs = 1
number_hidden_nodes = 120

nn_2 = Sequential()
nn_2.add(Dense(units=number_hidden_nodes, input_dim=number_inputs, activation="tanh"))
nn_2.add(Dense(units=1, activation="hard_sigmoid"))


# Compile model
nn_2.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# Fit the model
model_2 = nn_2.fit(X_train_scaled, y_train, epochs=100)


# Evaluate the model fit with linear dummy data
model_loss_2, model_accuracy_2 = nn_2.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Loss: {model_loss_2}, Accuracy: {model_accuracy_2}")
























y = df["pct change"]
X = df.drop(columns="pct change")


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, 
                                                   y, 
                                                   random_state=1, 
                                                   stratify=y)
X_train.shape






































df = aapl_complete


def window_data(df, window, feature_col_number, target_col_number):
    """
    This function accepts the column number for the features (X) and the target (y).
    It chunks the data up with a rolling window of Xt - window to predict Xt.
    It returns two numpy arrays of X and y.
    """
    X = []
    y = []
    for i in range(len(df) - window):
        features = df.iloc[i : (i + window), feature_col_number]
        target = df.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)





# Creating the features (X) and target (y) data using the window_data() function.
window_size = 5

feature_column = 6
target_column = 6
X, y = window_data(df, window_size, feature_column, target_column)
print (f"X sample values:\n{X[:5]} \n")
print (f"y sample values:\n{y[:5]}")



























#X = aapl_complete["Headline"].values
#y = aapl_sentiment["close"].values

















# Create the features set (X) and the target vector (y)
x_cols = [i for i in aapl_complete.columns if i not in ("pct change")]
X = aapl_complete[x_cols]
y = aapl_complete["pct change"]
X


# Create the train, test, and validation sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)


# Import the Tokenizer method from Keras
from tensorflow.keras.preprocessing.text import Tokenizer


# Create an instance of the Tokenizer and fit it with the X text data
#tokenizer = Tokenizer(lower=True)
#tokenizer.fit_on_texts(X)


# Print the first five elements of the encoded vocabulary
#for token in list(tokenizer.word_index)[:5]:
    #print(f"word: '{token}', token: {tokenizer.word_index[token]}")


# Transform the text data to numerical sequences
#X_seq = tokenizer.texts_to_sequences(X)


# Contrast a sample numerical sequence with its text version
#print("**Text comment**")
#print({X[1]})


#print("**Numerical sequence representation**")
#print(X_[0])


# Import the pad_sequences method from Keras
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Set the pad size
#max_words = 30

# Pad the sequences using the pad_sequences() method
#X_pad = pad_sequences(X_seq, maxlen=max_words, padding="post")


#print(X_pad)


# Creating training, validation, and testing sets using the encoded data
#X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(X_pad, y)

#X_train_rnn, X_val_rnn, y_train_rnn, y_val_rnn = train_test_split(X_train_rnn, y_train_rnn)


# Import Keras modules for model creation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


# Model set-up
#vocabulary_size = len(tokenizer.word_counts.keys()) + 1
#embedding_size = 64


# Define the LSTM RNN model
model = Sequential()

# Layer 1
#model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))

# Layer 2
model.add(LSTM(units=5))

# Output layer
model.add(Dense(units=1, activation="sigmoid"))


# Compile the model
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=[
        "accuracy",
        tf.keras.metrics.TruePositives(name="tp"),
        tf.keras.metrics.TrueNegatives(name="tn"),
        tf.keras.metrics.FalsePositives(name="fp"),
        tf.keras.metrics.FalseNegatives(name="fn"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)


# Show model summary
model.summary()


# Training the model
batch_size = 1000
epochs = 10
model.fit(
    X_train_rnn,
    y_train_rnn,
    validation_data=(X_val_rnn, y_val_rnn),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
)


# Predict classes using the testing data
y_rnn_pred = model.predict_classes(X_test_rnn, batch_size=1000)


# Accuracy
from sklearn.metrics import accuracy_score

print("RNN LSTM Accuracy get_ipython().run_line_magic(".2f"", " % (accuracy_score(y_test_rnn, y_rnn_pred)))")


# Import the confusion_matrix method from sklearn
from sklearn.metrics import confusion_matrix


# Confusion matrtix metrics from the RNN LSTM model
tn_rnn, fp_rnn, fn_rnn, tp_rnn = confusion_matrix(y_test_rnn, y_rnn_pred).ravel()

# Dataframe to display confusion matrix from the RNN LSTM model
cm_rnn_df = pd.DataFrame(
    {
        "Positive(1)": [f"TP={tp_rnn}", f"FP={fp_rnn}"],
        "Negative(0)": [f"FN={fn_rnn}", f"TN={tn_rnn}"],
    },
    index=["Positive(1)", "Negative(0)"],
)
cm_rnn_df.index.name = "Actual"
cm_rnn_df.columns.name = "Predicted"
print("Confusion Matrix from the RNN LSTM Model")
display(cm_rnn_df)


# Import the classification_report method from sklearn
from sklearn.metrics import classification_report


# Display classification report for the RNN LSTM Model
print("Classification Report for the RNN LSTM Model")
print(classification_report(y_rnn_pred, y_test_rnn))


# Import the roc_curve and auc metrics from sklearn
from sklearn.metrics import roc_curve, auc


# Making predictions to feed the roc_curve module
test_predictions_rnn = model.predict(X_test_rnn, batch_size=1000)


# Data for ROC Curve - RNN LSTM Model
fpr_test_rnn, tpr_test_rnn, thresholds_test_rnn = roc_curve(y_test_rnn, test_predictions_rnn)


# AUC for the RNN LSTM Model
auc_test_rnn = auc(fpr_test_rnn, tpr_test_rnn)
auc_test_rnn = round(auc_test_rnn, 4)


# Dataframe to plot ROC Curve for the RNN LSTM model
roc_df_test_rnn = pd.DataFrame({"FPR Test": fpr_test_rnn, "TPR Test": tpr_test_rnn,})


roc_df_test_rnn.plot(
    x="FPR Test",
    y="TPR Test",
    color="blue",
    style="--",
    xlim=([-0.05, 1.05]),
    title=f"Test ROC Curve (AUC={auc_test_rnn})",
)
