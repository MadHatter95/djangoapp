import csv
import numpy as np 
import pandas as pd 
#import PIL, PIL.Image, StringIO
import matplotlib.pyplot as plt
from matplotlib import style
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas 
from matplotlib.figure import Figure 
#import matplotlib matplotlib.use('Agg')
from matplotlib.dates import DateFormatter 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 
from matplotlib import pylab
from pylab import *

from pandas_datareader import data as pdr  
import fix_yahoo_finance as yf



class prediction():
    def fun(self):
        yf.pdr_override()

        style.use('ggplot')
        fig=plt.Figure() 
        #ax=fig.add_subplot(111) 
        #df = pdr.get_data_yahoo('TSLA', '2010-01-01','2018-03-20')
        df = pd.read_csv('C:\\DJANGO\\stock_market\\users\\csv\\TSLA.csv')
        df = df.dropna()
        df = df[['Open','High','Low','Close','Volume']]
        df.head()
        print(df.head())
        print(df.tail())
        df['Open-Close'] = df.Open - df.Close
        df['High-Low'] = df.High - df.Low
        df = df.dropna()
        X = df[['Open-Close','High-Low']]
        Y = df[['Volume']]
        Y.head()
        X.head()
        X.tail()
        print(X.head())
        print(X.tail())
        print(Y.head())

        Y= np.where(df['Close'].shift(-1)>df['Close'],1,-1)
        split_percentage = 0.7
        split = int(split_percentage*len(df))
        X_train = X[:split]
        Y_train = Y[:split]

        X_test = X[split:]
        Y_test = Y[split:]


        knn = KNeighborsClassifier(n_neighbors=15)

        knn.fit(X_train, Y_train)

        accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
        accuracy_test = accuracy_score(Y_test, knn.predict(X_test))

       # print('Train Data accuracy : %.2f' %accuracy_train)
       # print('Test Data accuracy : %.2f' %accuracy_test)

        df['Predicted_Signal'] = knn.predict(X)

        df['TSLA_Returns'] = np.log(df['Close']/df['Close'].shift(1))
        Cumulative_TSLA_Returns = df[split:]['TSLA_Returns'].cumsum()*100

        df['Strategy_Returns'] = df['TSLA_Returns'] * df['Predicted_Signal'].shift(1)
        Cum_Strategy_Returns = df[split:]['Strategy_Returns'].cumsum()*100
        c=df['Close']
        t=df['TSLA_Returns'].isnull()
        ap = float(sum(c))
        pp = float(sum(t))
        pre = (ap-pp)
        #plt.figure(figsize=(10,5))
        #c1=plt.plot(Cumulative_TSLA_Returns,color='r',label='Tesla Returns')
        #c2=plt.plot(Cum_Strategy_Returns,color='g',label='Strategy Returns')
        #plt.legend()
        #plt.show()

    
        return ap,pre



class predictapple():
    def funapple(self):
        yf.pdr_override()

        style.use('ggplot')
        #df = pdr.get_data_yahoo('TSLA', '2010-01-01','2018-03-20')
        df = pd.read_csv('C:\\DJANGO\\stock_market\\users\\csv\\AAPL.csv')
        df = df.dropna()
        df = df[['Open','High','Low','Close']]
        df.head()
        print(df.head())
        print(df.tail())
        df['Open-Close'] = df.Open - df.Close
        df['High-Low'] = df.High - df.Low
        df = df.dropna()
        X = df[['Open-Close','High-Low']]
        X.head()
        X.tail()
        print(X.head())
        print(X.tail())

        Y= np.where(df['Close'].shift(-1)>df['Close'],1,-1)
        split_percentage = 0.7
        split = int(split_percentage*len(df))
        X_train = X[:split]
        Y_train = Y[:split]

        X_test = X[split:]
        Y_test = Y[split:]


        knn = KNeighborsClassifier(n_neighbors=15)

        knn.fit(X_train, Y_train)

        accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
        accuracy_test = accuracy_score(Y_test, knn.predict(X_test))

       # print('Train Data accuracy : %.2f' %accuracy_train)
       # print('Test Data accuracy : %.2f' %accuracy_test)

        df['Predicted_Signal'] = knn.predict(X)

        df['AAPL_Returns'] = np.log(df['Close']/df['Close'].shift(1))
        Cumulative_AAPL_Returns = df[split:]['AAPL_Returns'].cumsum()*100

        df['Strategy_Returns'] = df['AAPL_Returns'] * df['Predicted_Signal'].shift(1)
        Cum_Strategy_Returns = df[split:]['Strategy_Returns'].cumsum()*100
        c=df['Close']
        t=df['AAPL_Returns'].isnull()
        ap = float(sum(c))
        pp = float(sum(t))
        pre = (ap-pp)
        #plt.figure(figsize=(10,5))
        #c1=plt.plot(Cumulative_AAPL_Returns,color='r',label='Apple Returns')
       # c2=plt.plot(Cum_Strategy_Returns,color='g',label='Strategy Returns')
        #plt.legend()
        #plt.show()

    
        return ap,pre

    # print(df['TSLA_Returns'])
     #   plt.figure(figsize=(10,5))
      #  plt.plot(Cumulative_TSLA_Returns,color='r',label='Tesla Returns')
       # plt.plot(Cum_Strategy_Returns,color='g',label='Strategy Returns')
        #plt.legend()
       # plt.show()
              


class predictairtel():
    def funairtel(self):
        yf.pdr_override()

        style.use('ggplot')
        #df = pdr.get_data_yahoo('TSLA', '2010-01-01','2018-03-20')
        df = pd.read_csv('C:\\DJANGO\\stock_market\\users\\csv\\BHARTIARTL.NS.csv')
        df = df.dropna()
        df = df[['Open','High','Low','Close']]
        df.head()
        print(df.head())
        print(df.tail())
        df['Open-Close'] = df.Open - df.Close
        df['High-Low'] = df.High - df.Low
        df = df.dropna()
        X = df[['Open-Close','High-Low']]
        X.head()
        X.tail()
        print(X.head())
        print(X.tail())

        Y= np.where(df['Close'].shift(-1)>df['Close'],1,-1)
        split_percentage = 0.7
        split = int(split_percentage*len(df))
        X_train = X[:split]
        Y_train = Y[:split]

        X_test = X[split:]
        Y_test = Y[split:]


        knn = KNeighborsClassifier(n_neighbors=15)

        knn.fit(X_train, Y_train)

        accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
        accuracy_test = accuracy_score(Y_test, knn.predict(X_test))

        #print('Train Data accuracy : %.2f' %accuracy_train)
        #print('Test Data accuracy : %.2f' %accuracy_test)

        df['Predicted_Signal'] = knn.predict(X)

        df['ARTL_Returns'] = np.log(df['Close']/df['Close'].shift(1))
        Cumulative_ARTL_Returns = df[split:]['ARTL_Returns'].cumsum()*100

        df['Strategy_Returns'] = df['ARTL_Returns'] * df['Predicted_Signal'].shift(1)
        Cum_Strategy_Returns = df[split:]['Strategy_Returns'].cumsum()*100
        c=df['Close']
        t=df['ARTL_Returns'].isnull()
        ap = float(sum(c))
        pp = float(sum(t))
        pre = (ap-pp)
       # plt.figure(figsize=(10,5))
       # c1=plt.plot(Cumulative_ARTL_Returns,color='r',label='Tesla Returns')
       # c2=plt.plot(Cum_Strategy_Returns,color='g',label='Strategy Returns')
       # plt.legend()
        #plt.show()

    
        return ap,pre

    # print(df['TSLA_Returns'])
     #   plt.figure(figsize=(10,5))
      #  plt.plot(Cumulative_TSLA_Returns,color='r',label='Tesla Returns')
       # plt.plot(Cum_Strategy_Returns,color='g',label='Strategy Returns')
        #plt.legend()
       # plt.show()
       # return accuracy_train , accuracy_test  



class predictgoogle():
    def fungoogle(self):
        yf.pdr_override()

        style.use('ggplot')
        #df = pdr.get_data_yahoo('TSLA', '2010-01-01','2018-03-20')
        df = pd.read_csv('C:\\DJANGO\\stock_market\\users\\csv\\GOOG.csv')
        df = df.dropna()
        df = df[['Open','High','Low','Close']]
        df.head()
        print(df.head())
        print(df.tail())
        df['Open-Close'] = df.Open - df.Close
        df['High-Low'] = df.High - df.Low
        df = df.dropna()
        X = df[['Open-Close','High-Low']]
        X.head()
        X.tail()
        print(X.head())
        print(X.tail())

        Y= np.where(df['Close'].shift(-1)>df['Close'],1,-1)
        split_percentage = 0.7
        split = int(split_percentage*len(df))
        X_train = X[:split]
        Y_train = Y[:split]

        X_test = X[split:]
        Y_test = Y[split:]


        knn = KNeighborsClassifier(n_neighbors=15)

        knn.fit(X_train, Y_train)

        accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
        accuracy_test = accuracy_score(Y_test, knn.predict(X_test))

        #print('Train Data accuracy : %.2f' %accuracy_train)
        #print('Test Data accuracy : %.2f' %accuracy_test)

        df['Predicted_Signal'] = knn.predict(X)

        df['GOOG_Returns'] = np.log(df['Close']/df['Close'].shift(1))
        Cumulative_GOOG_Returns = df[split:]['GOOG_Returns'].cumsum()*100

        df['Strategy_Returns'] = df['GOOG_Returns'] * df['Predicted_Signal'].shift(1)
        Cum_Strategy_Returns = df[split:]['Strategy_Returns'].cumsum()*100
        c=df['Close']
        t=df['GOOG_Returns'].isnull()
        ap = float(sum(c))
        pp = float(sum(t))
        pre = (ap-pp)
        #plt.figure(figsize=(10,5))
        #c1=plt.plot(Cumulative_GOOG_Returns,color='r',label='Tesla Returns')
        #c2=plt.plot(Cum_Strategy_Returns,color='g',label='Strategy Returns')
        #plt.legend()
        #plt.show()

    
        return ap,pre

    # print(df['TSLA_Returns'])
     #   plt.figure(figsize=(10,5))
      #  plt.plot(Cumulative_TSLA_Returns,color='r',label='Tesla Returns')
       # plt.plot(Cum_Strategy_Returns,color='g',label='Strategy Returns')
        #plt.legend()
       # plt.show()
       # return accuracy_train , accuracy_test              



class predictamazon():
    def funamzn(self):
        yf.pdr_override()

        style.use('ggplot')
        #df = pdr.get_data_yahoo('TSLA', '2010-01-01','2018-03-20')
        df = pd.read_csv('C:\\DJANGO\\stock_market\\users\\csv\\AMZN.csv')
        df = df.dropna()
        df = df[['Open','High','Low','Close']]
        df.head()
        print(df.head())
        print(df.tail())
        df['Open-Close'] = df.Open - df.Close
        df['High-Low'] = df.High - df.Low
        df = df.dropna()
        X = df[['Open-Close','High-Low']]
        X.head()
        X.tail()
        print(X.head())
        print(X.tail())

        Y= np.where(df['Close'].shift(-1)>df['Close'],1,-1)
        split_percentage = 0.7
        split = int(split_percentage*len(df))
        X_train = X[:split]
        Y_train = Y[:split]

        X_test = X[split:]
        Y_test = Y[split:]


        knn = KNeighborsClassifier(n_neighbors=15)

        knn.fit(X_train, Y_train)

        accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
        accuracy_test = accuracy_score(Y_test, knn.predict(X_test))

        #print('Train Data accuracy : %.2f' %accuracy_train)
        #print('Test Data accuracy : %.2f' %accuracy_test)

        df['Predicted_Signal'] = knn.predict(X)

        df['AMZN_Returns'] = np.log(df['Close']/df['Close'].shift(1))
        Cumulative_AMZN_Returns = df[split:]['AMZN_Returns'].cumsum()*100

        df['Strategy_Returns'] = df['AMZN_Returns'] * df['Predicted_Signal'].shift(1)
        Cum_Strategy_Returns = df[split:]['Strategy_Returns'].cumsum()*100
        c=df['Close']
        t=df['AMZN_Returns'].isnull()
        ap = float(sum(c))
        pp = float(sum(t))
        pre = (ap-pp)
        #plt.figure(figsize=(10,5))
        #c1=plt.plot(Cumulative_AMZN_Returns,color='r',label='Tesla Returns')
        #c2=plt.plot(Cum_Strategy_Returns,color='g',label='Strategy Returns')
        #plt.legend()
        #plt.show()

    
        return ap,pre

    # print(df['TSLA_Returns'])
     #   plt.figure(figsize=(10,5))
      #  plt.plot(Cumulative_TSLA_Returns,color='r',label='Tesla Returns')
       # plt.plot(Cum_Strategy_Returns,color='g',label='Strategy Returns')
        #plt.legend()
       # plt.show()
       # return accuracy_train , accuracy_test        



class predictmicrosoft():
    def funmsft(self):
        yf.pdr_override()

        style.use('ggplot')
        #df = pdr.get_data_yahoo('TSLA', '2010-01-01','2018-03-20')
        df = pd.read_csv('C:\\DJANGO\\stock_market\\users\\csv\\MSFT.csv')
        df = df.dropna()
        df = df[['Open','High','Low','Close']]
        df.head()
        print(df.head())
        print(df.tail())
        df['Open-Close'] = df.Open - df.Close
        df['High-Low'] = df.High - df.Low
        df = df.dropna()
        X = df[['Open-Close','High-Low']]
        X.head()
        X.tail()
        print(X.head())
        print(X.tail())

        Y= np.where(df['Close'].shift(-1)>df['Close'],1,-1)
        split_percentage = 0.7
        split = int(split_percentage*len(df))
        X_train = X[:split]
        Y_train = Y[:split]

        X_test = X[split:]
        Y_test = Y[split:]


        knn = KNeighborsClassifier(n_neighbors=15)

        knn.fit(X_train, Y_train)

        accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
        accuracy_test = accuracy_score(Y_test, knn.predict(X_test))

        #print('Train Data accuracy : %.2f' %accuracy_train)
        #print('Test Data accuracy : %.2f' %accuracy_test)

        df['Predicted_Signal'] = knn.predict(X)

        df['MSFT_Returns'] = np.log(df['Close']/df['Close'].shift(1))
        Cumulative_MSFT_Returns = df[split:]['MSFT_Returns'].cumsum()*100

        df['Strategy_Returns'] = df['MSFT_Returns'] * df['Predicted_Signal'].shift(1)
        Cum_Strategy_Returns = df[split:]['Strategy_Returns'].cumsum()*100
        c=df['Close']
        t=df['MSFT_Returns'].isnull()
        ap = float(sum(c))
        pp = float(sum(t))
        pre = (ap-pp)
        #plt.figure(figsize=(10,5))
        #c1=plt.plot(Cumulative_MSFT_Returns,color='r',label='Tesla Returns')
        #c2=plt.plot(Cum_Strategy_Returns,color='g',label='Strategy Returns')
        #plt.legend()
        #plt.show()

    
        return ap,pre

    # print(df['TSLA_Returns'])
     #   plt.figure(figsize=(10,5))
      #  plt.plot(Cumulative_TSLA_Returns,color='r',label='Tesla Returns')
       # plt.plot(Cum_Strategy_Returns,color='g',label='Strategy Returns')
        #plt.legend()
       # plt.show()
      #  return accuracy_train , accuracy_test      



class predictsensex():
    def funsensex(self):
        yf.pdr_override()

        style.use('ggplot')
        #df = pdr.get_data_yahoo('TSLA', '2010-01-01','2018-03-20')
        df = pd.read_csv('C:\\DJANGO\\stock_market\\users\\csv\\SENSEX.csv')
        df = df.dropna()
        df = df[['Open','High','Low','Close']]
        df.head()
        print(df.head())
        print(df.tail())
        df['Open-Close'] = df.Open - df.Close
        df['High-Low'] = df.High - df.Low
        df = df.dropna()
        X = df[['Open-Close','High-Low']]
        X.head()
        X.tail()
        print(X.head())
        print(X.tail())

        Y= np.where(df['Close'].shift(-1)>df['Close'],1,-1)
        split_percentage = 0.7
        split = int(split_percentage*len(df))
        X_train = X[:split]
        Y_train = Y[:split]

        X_test = X[split:]
        Y_test = Y[split:]


        knn = KNeighborsClassifier(n_neighbors=15)

        knn.fit(X_train, Y_train)

        accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
        accuracy_test = accuracy_score(Y_test, knn.predict(X_test))

        #print('Train Data accuracy : %.2f' %accuracy_train)
        #print('Test Data accuracy : %.2f' %accuracy_test)

        df['Predicted_Signal'] = knn.predict(X)

        df['SENSEX_Returns'] = np.log(df['Close']/df['Close'].shift(1))
        Cumulative_SENSEX_Returns = df[split:]['SENSEX_Returns'].cumsum()*100

        df['Strategy_Returns'] = df['SENSEX_Returns'] * df['Predicted_Signal'].shift(1)
        Cum_Strategy_Returns = df[split:]['Strategy_Returns'].cumsum()*100
        c=df['Close']
        t=df['SENSEX_Returns'].isnull()
        ap = float(sum(c))
        pp = float(sum(t))
        pre = (ap-pp)
        #plt.figure(figsize=(10,5))
        #c1=plt.plot(Cumulative_SENSEX_Returns,color='r',label='Tesla Returns')
        #c2=plt.plot(Cum_Strategy_Returns,color='g',label='Strategy Returns')
        #plt.legend()
        #plt.show()

    
        return ap,pre

    # print(df['TSLA_Returns'])
     #   plt.figure(figsize=(10,5))
      #  plt.plot(Cumulative_TSLA_Returns,color='r',label='Tesla Returns')
       # plt.plot(Cum_Strategy_Returns,color='g',label='Strategy Returns')
        #plt.legend()
       # plt.show()
       # return accuracy_train , accuracy_test          