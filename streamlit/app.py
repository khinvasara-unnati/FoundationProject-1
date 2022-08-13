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
    st.markdown('''### Reliance Company Info ''')
    
    df = pd.read_csv('https://raw.githubusercontent.com/gitsim02/FoundationProject-1/main/data/reliance_prices.csv')
    col1, col2 = st.columns(2)
    col2.markdown(" ###### Reliance Price Dataset with feature engineering")
    col2.write(df.head(3))
    
    col1.markdown("""We have extracted Reliance Prices from 13-Dec-2016 using Yahoo Finance API.  
    Reliance Industries Limited (RIL) is one of the pioneer companies in India, with a history of over 40 years. 
    It has diverse businesses including energy, petrochemicals, natural gas, retail, telecommunications, mass media, and textiles. 
    RIL stock is consistently ranked among the Best Long Term Stock to buy in the Indian Markets. 
    The stock has ranged from minimum of Rs 20 in 2000s to its current price of Rs.2600+ 
    Click on the link to view the Reliance Stock Price data:   
    https://raw.githubusercontent.com/gitsim02/FoundationProject-1/main/data/reliance_prices.csv    
    """)
  

plots = st.container()
with plots:
    st.markdown(""" ##### Reliance Price Trend Chart over 5 Years""")
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



import yfinance as yf
GetRILInformation = yf.Ticker("Reliance.NS")
from datetime import date, timedelta

stock = 'Reliance.NS'
interval = '1d'
today = date.today()
yesterday = today - timedelta(20)

latest_price = yf.download(tickers=stock, period='1d', interval='1m')['Close'][0]

businesscriteria = st.container()
with businesscriteria:
    st.markdown(" ### Economic Success Criteria")
    col1, col2 = st.columns(2)
    col1.markdown(""" ##### Why you should choose this model over others:   
    Let's say you want to invest in Reliance over a longer horizon  
    (as opposed to investing at one point so as to mitigate risk)  
    Our extensively trained model will suggest to Buy or Hold this stock based on two parameters  
    1. Last 10 prices  
    2. Sentiment Score of any recent news      
    This model has been tested against a data of 392 days (from 9-Apr-2021, till now)               
    """)
    col2.markdown(""" ##### How the Model works:  
    1. We used Long Short Term Model(LSTM) to predict next day's closing price.  
    2. Along with that, we used Ordinary Least Squares(OLS) to regress against the residuals.  
    3. With this, our model gains a fit of 99%.  
    4. LTSM Model has an RMSE error of Rs.50.  
    5. After regressing the error with news sentiment, the RMSE further reduced to Rs.6.""")

    st.write("If you followed this model to decide your investment on your buy strategy, you would possess 25 stocks amounting to Rs.50,590")
    st.write("The price of 25 RIL stocks now is:", latest_price * 25, "\n",
             "Return on Investment comes to:", round((latest_price * 25 * 100 / 50590) - 100, 2), "%")
# For Purchase price of 50590, refer LSTM_OLS_Model, cell 61-62
# This is sum of Share Price of all dates where model recommends Buy stance

def get_stock_data(stock,startdate,enddate,interval):
        ticker = stock  
        yf.pdr_override()
        df = yf.download(tickers=stock, start=startdate, end=enddate, interval=interval)
        df.reset_index(inplace=True) 
        df['Date'] = df['Date'].dt.date
        return df


financedata = st.container()
with financedata:

    st.markdown(""" #### Last 10 days Stock Price""")

    col1, col2 = st.columns(2)

    stock_data = get_stock_data(stock,yesterday,today,interval)
    last10prices = stock_data[['Date','Close']].tail(10)
    #last10prices.set_index("Date",inplace=True)
    col1.write(last10prices)

    last10prices.plot(kind="line", x="Date", y="Close", figsize=(10,5))
    plt.ylabel('Price', fontsize=10)
    plt.xlabel('Time Period', fontsize=10)
    plt.title('Price Trend of Last 10 Days', fontsize=12)
    plt.show()
    col2.pyplot(fig=plt, use_container_width=True)
    st.write("Latest Price: ",latest_price)

textinput = st.container()
with textinput:
    col1, col2 = st.columns(2)
    text = col1.text_input("Enter the most recent news article to obtain sentiment score and respective price prediction",
                         "Example: Reliance recorded high profits for Quarter 3")
    vader_score = sentAnalyzer.polarity_scores(text)['compound']
    with col2:
        st.markdown("  "
                    "  ")
        st.write(" The sentiment score of the given document is:",vader_score)

import urllib.request
from keras.models import load_model

# urllib.request.urlretrieve(
#     'https://github.com/gitsim02/FoundationProject-1/tree/main/lstm_model',
#     'lstm_model')

lstm_model = load_model("lstm_model")

prediction = st.container()
with prediction:
    st.markdown("###### Basis the above share price, and the sentiment of the input news article, "
                "our model predict the next day's price to be:")
    # Incorporating the news sentiment score
    lstm_predicted_price = lstm_model.predict([last10prices.Close.tolist()])

    ols_adjusted_pred = lstm_predicted_price + 6.475 - 14.88 * vader_score

    st.write(ols_adjusted_pred[0][0])

    if ols_adjusted_pred>1.003*latest_price:
        st.write("As the next day's price is estimated to be 30bps more than today, it is recommended to Buy")
    else:
        st.write("As there is no significant change, it is recommended to Hold")

end = time.time()

st.write("Performance - Time taken to execute the entire program:",round(end-start,4),"seconds")