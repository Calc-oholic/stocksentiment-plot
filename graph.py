import time

import yfinance as yf
from GoogleNews import GoogleNews
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import nltk

nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()
news = GoogleNews(lang='en')

def fetch_news(ticker, date):
    news_date = date.strftime('%m/%d/%Y')
    next_day = date + timedelta(days=1)
    next_day_str = next_day.strftime('%m/%d/%Y')
    news.clear()
    news.set_time_range(news_date, next_day_str)
    news.get_news(f'{ticker}')
    return news.results()

# def fetch_news(ticker, date):
#     news_date = date.strftime('%m/%d/%Y')
#     news.clear()
#     news.search(f'{ticker} {news_date}')
#     return news.results()

def calculate_sentiment(xnews):
    scores = [analyzer.polarity_scores(article['title'])['compound'] for article in xnews]
    return np.mean(scores) if scores else 0

def fetch_stock_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

def build_sentiment_dataset(ticker, days):
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    sentiment_scores = []
    dates = []

    for i in range(1, len(stock_data)):
        date = stock_data.index[i - 1].date()
        next_date = stock_data.index[i].date()

        xnews = fetch_news(ticker, date)
        sentiment = calculate_sentiment(xnews)
        sentiment_scores.append(sentiment)
        dates.append(next_date.strftime('%m/%d'))

    stock_data = stock_data[1:]
    return pd.DataFrame(
        {'Date': dates, 'Sentiment': sentiment_scores, 'Price Change': stock_data['Adj Close'].pct_change().values})


def plot_sentiment_vs_price(df):
    fig, sent = plt.subplots()

    sent.set_xlabel('Date')
    sent.set_ylabel('Sentiment Score', color='tab:blue')
    sent.plot(df['Date'], df['Sentiment'], color='tab:blue', label="Sentiment")
    sent.tick_params(axis='y', labelcolor='tab:blue')

    change = sent.twinx()
    change.set_ylabel('% Price Change', color='tab:red')
    change.plot(df['Date'], df['Price Change'], color='tab:red', label="Price Change")
    change.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title(f"{ticker} {days}")
    plt.savefig('plot.png')
    plt.show()


ticker = 'HOG'
days = 100

df = build_sentiment_dataset(ticker, days)
plot_sentiment_vs_price(df)
