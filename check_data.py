#MarketBasketAnalysis

#import libraries
import numpy as np 
import pandas as pd 
from mlxtend.preprocessing import TransactionEncoder 

#Data visualization
import matplotlib.pyplot as plt
import seaborn as sb
import squarify

# for market basket analysis
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

#setting
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#read the data
store = pd.read_csv('/Users/nadinekathi/Documents/GitHub/MarketBasketAnalysis/GroceryStoreDataSet.csv',names=['product'],header=None)


#check data set
print(store.shape) #(50,1)

#checkng the head, tail and random part of the data set
print(store.head())
print(store.tail())
print(store.sample(3))

#Statistical description of the dataset.
print(store.describe())

#Data Visualizations

#data set prep, split column in several columns
store_df = store['product'].str.split(",", expand=True)


#bar chart
# looking at the frequency of most popular items 
plt.rcParams['figure.figsize'] = (18, 7)
color = plt.cm.copper(np.linspace(0, 1, 40))
store_df[0].value_counts().head(40).plot.bar(color = color)
plt.title('frequency of most popular items', fontsize = 20)
plt.xticks(rotation = 90 )
plt.grid()
plt.show()

# plotting a tree map

y = store_df[0].value_counts().head(50).to_frame()
y.index

plt.rcParams['figure.figsize'] = (20, 20)
squarify.plot(sizes = y.values, label = y.index, 
              alpha=.8, 
              color = sb.color_palette("magma"), 
              ec = 'white')
plt.title('Tree Map for Popular Items')
plt.axis('off')
plt.show()

#popular items are: bread, milk, coffee etc.


# create list
store_list = list(store['product'].apply(lambda x: x.split(",")))

# 1 transaction per row with each product 1 hot encoded
te = TransactionEncoder()
store_ap = te.fit(store_list).transform(store_list)
store_ap = pd.DataFrame(store_ap,columns=te.columns_)

# checking the shape
print(store_ap.shape) # [50 rows x 29 columns]
print(store.head)


# Using Apriori Algorithm


#Create some rules

frequent_itemsets = apriori(store_ap, min_support=0.07, use_colnames=True) #support higher (relative frequency that the rules show up)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets)

# Metric can be set to confidence, lift, support, leverage and conviction.
#using lift
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.3) #(0.3 / 0.8)
rules["antecedents_length"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequents_length"] = rules["consequents"].apply(lambda x: len(x))
rules.sort_values("lift")
rules.head()
print(rules)

#using Confidence
rules2 = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)#.iloc[:,:-3]
rules2["antecedents_length"] = rules2["antecedents"].apply(lambda x: len(x))
rules2["consequents_length"] = rules2["consequents"].apply(lambda x: len(x))
rules2.sort_values("confidence")
print(rules2) #.head()


# filter the dataframe using standard pandas code
# look for a large lift (2) and high confidence (.6):

print(rules2.loc[(rules2['lift']>=2) & (rules2['confidence']>= 0.6)] )


#Visualizing results
#1.Support vs Confidence


plt.scatter(rules2['support'], rules2['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

#2. Support vs Lift

plt.scatter(rules2['support'], rules2['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.show()

