# random GroceryStoreDataSet.csv file

import numpy as np # linear algebra
import pandas as pd #read csv.file


#read the data
store=pd.read_csv('/Users/nadinekathi/Documents/GitHub/MarketBasketAnalysis/GroceryStoreDataSet.csv',names=['product'],header=None)

#check data set
print(store.shape) #(50,1)

#checkng the head, tail and random part of the data set
print(store.head())
#print(store.tail())
#print(store.random(10))
#Statistical description of the dataset.
#print(store.describe())

#prep the data set

#store=list(store['product'].apply(lambda x: x.split(",")))

#Preprocesing of Data
#print(store)
