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
import yfinance as yf       

# SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND PAGE ICON
st.set_page_config(layout="centered", page_icon=":taxi:")

logo = st.container()
header = st.container()
dataset = st.container()
plots = st.container()
textinput = st.container()

with logo:
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    #background = st.image("https://github.com/Kenrich005/Uber_reviews_textanalytics/blob/6ace3968785f4cfcaca57fbc589238935940bc86/ISB_Logo_JPEG.jpg?raw=true")
    col6.image("https://github.com/Kenrich005/Uber_reviews_textanalytics/blob/6ace3968785f4cfcaca57fbc589238935940bc86/ISB_Logo_JPEG.jpg?raw=true", width=200)

with header:
    
    st.title('Application to Predict Stock Prices using sentiment analysis of Financial news')
    st.markdown("""---""")
    st.markdown("""Kenny Devarapalli  
    Unnati Khinvasara  
    Raktim Prakash Srivastava  
    Siddharth Maheshwari  
    Anjali Rathore  """)
 
 

with dataset:
    st.markdown('''Reliance Prices dataset  
    We have extracted Reliance Prices from 13-Dec-2016 using Yahoo Finance API
    Click on the link to see the file: https://raw.githubusercontent.com/gitsim02/FoundationProject-1/main/reliance_prices.csv''')
    
    df = pd.read_csv('https://raw.githubusercontent.com/gitsim02/FoundationProject-1/main/reliance_prices.csv')
    st.write(df.head(5))
    
with plots:
    df.plot(kind = "line",x="Date",y=["Close","EMA30","SMA30"],figsize=(10,5) )
    plt.ylabel('Price', fontsize=12)
    plt.xlabel('Time Period', fontsize=12)
    plt.title('Price Distribution over 5 Years', fontsize = 12)
    plt.show()
    st.pyplot(fig=plt, use_container_width=False)
    
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sentAnalyzer = SentimentIntensityAnalyzer()

with textinput:
    text = st.text_input("Enter the most recent new article", "")
    vader_score = sentAnalyzer.polarity_scores(text)['compound']
    st.write("The sentiment score of the given document is:",vader_score)
    
GetRILInformation = yf.Ticker("Reliance.NS")

financedata = st.container()

from datetime import date, timedelta



stock = 'Reliance.NS'
interval = '1d'
today = date.today()
yesterday = today - timedelta(5)



def get_stock_data(stock,startdate,enddate,interval):
        ticker = stock  
        yf.pdr_override()
        df = yf.download(tickers=stock, start=startdate, end=enddate, interval=interval)
        df.reset_index(inplace=True)
        df['Date'] = df['Date'].dt.date
      
        return df
        financedata = st.container()



with financedata:
    # display Company Sector
    st.write("Company Sector : ", GetRILInformation.info['sector'])
     
    # display Price Earnings Ratio
    st.write("Price Earnings Ratio : ", GetRILInformation.info['trailingPE'])
     
    # display Company Beta
    st.write(" Company Beta : ", GetRILInformation.info['beta'])
  
    stock_data = get_stock_data(stock,yesterday,today,interval)
    st.write(stock_data)
    
    current_data = yf.download(tickers=stock, period='1d', interval='1m')['Close'][0]
    st.write("Latest Price: ",round(current_data,4))    