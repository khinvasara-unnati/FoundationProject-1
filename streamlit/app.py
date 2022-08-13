# Kenny Style
import time
start = time.time()

import streamlit as st

# Initializing all libraries
import pandas as pd
import numpy as np

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px               
from plotly.subplots import make_subplots 
import plotly.graph_objects as go

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sentAnalyzer = SentimentIntensityAnalyzer()

# Setting Page Config to Centered
st.set_page_config(page_title = "Application to Predict Stock Prices using sentiment analysis of Financial news",layout="wide")

logo = st.container()
with logo:
   col1, col2 = st.columns([4, 1])
   col1.title("Stock Price prediction using Sentiment Analysis")
   col2.image("https://github.com/gitsim02/FoundationProject-1/blob/377e2fb72eb68f2d7559fb8c0c617ca89d8db845/transparent_Black.jpg?raw=true",  width=150)
   st.markdown("""------""")
  
    
dataset = st.container()
with dataset:
    st.markdown('''### Reliance Prices dataset ''')
    
    df = pd.read_csv('https://raw.githubusercontent.com/gitsim02/FoundationProject-1/main/data/reliance_prices.csv')
    col1, col2 = st.columns(2)
    col2.write(df.head(5))
    
    col1.markdown("""We have extracted Reliance Prices from 13-Dec-2016 using Yahoo Finance API  
    Click on the link to see the file:   
    https://raw.githubusercontent.com/gitsim02/FoundationProject-1/main/data/reliance_prices.csv
    
    """)
  

plots = st.container()
with plots:
    st.markdown(""" ### Reliance Price Trend Chart over 5 Years""")
    selection = st.multiselect('Choose your preferred features for plotting',list(df.columns),['Close','EMA30','SMA30'])    
    selected_list = pd.DataFrame(selection,columns=["Selection"])["Selection"].tolist()
    
    if len(selected_list)==0:
        selected_list = ['Close','EMA30','SMA30']
    
    df.plot(kind = "line",x="Date",y=selected_list,figsize=(20,3) )
    plt.ylabel('Price', fontsize=10)
    plt.xlabel('Time Period', fontsize=10)
    plt.title('Price Distribution over 5 Years', fontsize = 12)
    plt.show()
    st.pyplot(fig=plt, use_container_width=False)
    st.write("\n\n\n")
    st.markdown("""        """)


st.markdown(""" ### Stock Prediction model using LSTM and News Sentiment Analysis  """)

plot2 = st.container()
with plot2:
    df_lstm = pd.read_csv("https://raw.githubusercontent.com/gitsim02/FoundationProject-1/main/data/final_dataset.csv")
    df_lstm[['Close', 'Predictions', 'ols_adjusted_pred']].plot(kind='line', figsize=(20,3))
    plt.ylabel('Price', fontsize=10)
    plt.xlabel('Time Period', fontsize=10)
    plt.title('Price Distribution over 5 Years', fontsize=12)
    plt.show()
    st.pyplot(fig=plt, use_container_width=False)
    st.write("\n\n\n")
    st.markdown(""" We used Long Short Term Model to predict next day's closing price.   
       - Along with that, we used Ordinary Least Squares Model to regress against the residuals.  
       - With this, our model gains a fit of 99%.  
       - LTSM Model has an RMSE error of Rs.50.
       - After regressing the error with news sentiment, the RMSE further reduced to Rs.6.""")


businesscriteria = st.container()
with businesscriteria:
    st.markdown(" ### Business and Economic Success Criteria")
    st.markdown(""" Let's say you want to invest in Reliance over a longer horizon (as opposed to investing at one point)
    
    """)



def get_stock_data(stock,startdate,enddate,interval):
        ticker = stock  
        yf.pdr_override()
        df = yf.download(tickers=stock, start=startdate, end=enddate, interval=interval)
        df.reset_index(inplace=True) 
        df['Date'] = df['Date'].dt.date
        return df


import yfinance as yf
GetRILInformation = yf.Ticker("Reliance.NS")
from datetime import date, timedelta

stock = 'Reliance.NS'
interval = '1d'
today = date.today()
yesterday = today - timedelta(20)

financedata = st.container()
with financedata:
    st.markdown(""" #### Last 10 days Stock Price""")
      
    stock_data = get_stock_data(stock,yesterday,today,interval)
    last10prices = stock_data[['Date','Close']].tail(10)
    st.write(last10prices)
    
    current_data = yf.download(tickers=stock, period='1d', interval='1m')['Close'][0]
    st.write("Latest Price: ",current_data)
    
textinput = st.container()
with textinput:
    text = st.text_input("Enter the most recent news article ", "")
    vader_score = sentAnalyzer.polarity_scores(text)['compound']
    st.write("The sentiment score of the given document is:",vader_score)

import urllib.request
from keras.models import load_model

# urllib.request.urlretrieve(
#     'https://github.com/gitsim02/FoundationProject-1/tree/main/lstm_model',
#     'lstm_model')

lstm_model = load_model("lstm_model")

prediction = st.container()
with prediction:
    st.markdown("#### Given the last 10 prices, and the recent news sentiment, our model predict the next days price to be:")
    # Incorporating the news sentiment score
    ols_adjusted_pred = lstm_model.predict([last10prices.Close.tolist()])[0] + 6.475 - 14.88 * vader_score
    st.write(ols_adjusted_pred)

end = time.time()

st.write("Time taken to execute the program:",round(end-start,4),"seconds")