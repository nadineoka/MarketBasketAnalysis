# MarketBasketAnalysis

Market basket analysis is a technique used by businesses to identify associations between products or services that are frequently purchased together. It is a type of data mining that involves analyzing customer transaction data, such as point-of-sale records or e-commerce shopping carts, to find patterns in customer buying behavior.

The set of items a customer buys is referred to as an <i>itemset</i>, and market basket analysis seeks to find relationships between purchases.

For this analysis we are going to use the Apriori algorithm. The Apriori algorithm works by first identifying all item sets that have a support greater than or equal to a specified threshold. The support of an item set is the proportion of transactions in which the item set appears.

Once the item sets with sufficient support have been identified, the algorithm generates new candidate item sets by combining these item sets with other item sets that have sufficient support. This process is repeated until no more item sets with sufficient support can be generated.

#### The Apriori algorithm can be used to:

<ul>
  <li>Identify frequent item sets in a transaction database.</li>
  <li>Determine the association rules between items, including which items tend to be purchased together and which items tend to be purchased separately.</li>
  <li>Determine the minimum support and minimum confidence levels needed for the association rules to be considered significant.</li>
  <li>Identify which items should be placed near each other in a store or online store to encourage customers to purchase related items.</li>
</ul>

#### Calculations

We want to calculate the support, confidence, and lift for the association rule {A, B} => {C}, which means "if a customer buys items A and B together, they are likely to buy item C as well."

Support:
The support measures the frequency of occurrence of a particular item set in the transaction dataset.
Support ({A, B}) = Number of transactions containing {A, B} / Total number of transactions

Confidence:
The confidence measures the probability that item C is purchased given that items A and B are purchased together.
Confidence ({A, B} => {C}) = Support ({A, B, C}) / Support ({A, B})

Lift:
The lift measures the strength of the association between item sets. A lift value greater than 1 indicates a positive association between the item sets, while a value less than 1 indicates a negative association.
Lift ({A, B} => {C}) = Support ({A, B, C}) / (Support ({A, B}) x Support ({C}))


# Getting started

### Importing the required libraries

```py
#for basic operations
import numpy as np 
import pandas as pd 
from mlxtend.preprocessing import TransactionEncoder 

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sb
import squarify

# for market basket analysis
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
```

```py
#setting
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```

### Importing the dataset

```py
#read the data
store = pd.read_csv('~/MarketBasketAnalysis/GroceryStoreDataSet.csv',names=['product'],header=None)

#check data set
print(store.shape) 
```

<i> 50 rows, 1 column</i>

```py
#Statistical description of the dataset.
print(store.describe())
```

<table>
<tr>
    <th>product count</th>
    <td>50</td>
</tr>
<tr>
    <th>unique</th>
    <td>47</td>
</tr>
<tr>
    <th>top</th>
    <td>BREAD,COFFEE,SUGAR</td>
</tr>
<tr>
    <th>freq</th>
    <td>2</td>
</tr>
</table>
 

```py 
#show the first few rows
print(store.head())
```

<table>
<tr>
    <th></th>
    <th>product</th>
<tr>
    <td>0</td>
    <td>MILK,BREAD,BISCUIT</td>
</tr>
<tr>
    <td>1</td>
    <td>PIZZA</td>
</tr>
<tr>
    <td>2</td>
    <td>PIZZA,WATER,BEER</td>
</tr>
</table>

```py 
#checking random entries in the dataset
print(store.random(3))
```

<table>
<tr>
    <th></th>
    <th>product</th>
</tr>
    <td>8</td>
    <td>SALT,JUICE,COFFEE</td>
</tr>
<tr>
    <td>45</td>
    <td>BREAD,SUGAR,BOURNVITA</td>
</tr>
<tr>
    <td>15</td>
    <td>SHRIMP,EGGS,AVOCADO,BREAD</td>
</tr>
</table>


# Data Visualizations

```py 
#data set prep, split column in several columns
store_df = store['product'].str.split(",", expand=True)
```

```py 
#show the first few rows
print(store_df.head())
```
<table>
<tr>
    <td></td>
    <td>0</td>
    <td>1</td>
    <td>2</td>
    <td>3</td>
    <td>4</td>
</tr>
<tr>
    <td>0</td>
    <td>MILK</td>
    <td>BREAD</td>
    <td>BISCUIT</td>
    <td>None</td>
    <td>None</td>
</tr>
<tr>
    <td>1</td>
    <td>PIZZA</td>
    <td>None</td>
    <td>None</td>
    <td>None</td>
    <td>None</td>
</tr>
<tr>
    <td>2</td>
    <td>PIZZA</td>
    <td>WATER</td>
    <td>BEER</td>
    <td>None</td>
    <td>None</td>
</tr>
</table>

```py
#Statistical description of the dataset.
print(store_df.describe())
```
<table>
<tr>
    <td></td>
    <td>0</td>
    <td>1</td>
    <td>2</td>
    <td>3</td>
    <td>4</td>
</tr>
<tr>
    <td>count</td>
    <td>50</td>
    <td>49</td>
    <td>45</td>
    <td>16</td>
    <td>1</td>
</tr>
<tr>
    <td>unique</td>
    <td>20</td>
    <td>18</td>
    <td>15</td>
    <td>10</td>
    <td>1</td>
</tr>
<tr>
    <td>top</td>
    <td>BREAD</td>
    <td>BREAD</td>
    <td>BISCUIT</td>
    <td>CORNFLAKES</td>
    <td>CORNFLAKES</td>
</tr>
<tr>
    <td>freq</td>
    <td>9</td>
    <td>6</td>
    <td>8</td>
    <td>4</td>
    <td>1</td>
</tr>
</table>


### bar chart
```py
# looking at the frequency of most popular items 
plt.rcParams['figure.figsize'] = (18, 7)
color = plt.cm.copper(np.linspace(0, 1, 40))
store_df[0].value_counts().head(40).plot.bar(color = color)
plt.title('frequency of most popular items', fontsize = 20)
plt.xticks(rotation = 90 )
plt.grid()
plt.show()
```

### plotting a tree map

```py
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
```
result: Bread and Coffee being the most frequent item on the list


# Apriori algorithm

This analysis requires that all the data for a transaction be included in 1 row and the items should be 1-hot encoded.

```py 
#create list
store_list = list(store['product'].apply(lambda x: x.split(",")))
```
['MILK', 'BREAD', 'BISCUIT'], ['PIZZA'], ['PIZZA', 'WATER', 'BEER']

```py 
# 1 transaction per row with each product 1 hot encoded
te = TransactionEncoder()
store_ap = te.fit(store_list).transform(store_list)
store_ap = pd.DataFrame(store_ap,columns=te.columns_)
```

```py 
print(store.head())
```
<table>
<tr>
    <td></td>
    <td>APPLE</td>
    <td>AVOCADO</td>
    <td>BANANA</td>
    <td>BEER</td>
    <td>...</td>
</tr>
<tr>
    <td>0</td>
    <td>False</td>
    <td>False</td>
    <td>False</td>
    <td>False</td>
    <td>...</td>
</tr>
<tr>
    <td>1</td>
    <td>False</td>
    <td>False</td>
    <td>False</td>
    <td>False</td>
    <td>...</td>
</tr>
<tr>
    <td>2</td>
    <td>False</td>
    <td>False</td>
    <td>False</td>
    <td>True</td>
    <td>...</td>
</tr>
</table>

```py 
print(store_ap.shape)
```
[50 rows x 29 columns]

 
#### Create some rules

The algorithm employs level-wise search for frequent itemsets. A list of all possible itemsets is generated with having a support value greater than min_support value = 0.07

```py
frequent_itemsets = apriori(store_ap, min_support=0.07, use_colnames=True) #support higher (relative frequency that the rules show up)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets)
```

<table>
<tr>
    <th></th>
    <th>support</th>
    <th>itemsets</th>
    <th>length</th>
</tr>
<tr>
    <td>0</td>
    <td>0.12</td>
    <td>(AVOCADO)</td>
    <td>1</td>
</tr>
<tr>
    <td>1</td>
    <td>0.22</td>
    <td>(BISCUIT)</td>
    <td>1</td>
</tr>
</table>
...

Typically, <b>support</b> is used to measure the abundance or frequency (often interpreted as significance or importance).
We refer to an itemset as a "frequent itemset" if the support is larger than a specified minimum-support threshold. 
Next, we will generate the rules with their corresponding support, confidence and lift. <b>Lift</b> is the ratio of the observed support to that expected if the two rules were independent and <b>confidence</b> is a measure of the reliability of the rule.

The <b>Min_support</b> is a floating point value between 0 and 1 that indicates the minimum support required for an itemset to be selected.
-> number of observation with item / total observation

The <b>antecedents</b> refers to the set of items that are used to predict or recommend another set of items in a customer's transaction history. The <b>consequent</b> is a term used to refer to the item or items that are being predicted or recommended based on the presence of another set of items in a customer's transaction history. In short, the antecedents refers to the item already bought, the consequent refers to the possible purchase. Antecedent support refers to the frequency of antecedent and Consequent support refers to the frequency of consequent.

Metric can be set to confidence, lift, support, leverage and conviction:

```py
#lift
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.3) 
rules["antecedents_length"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequents_length"] = rules["consequents"].apply(lambda x: len(x))
rules.sort_values("lift")
```

```py
rules.head()
```

<table>
<tr>
    <td></td>
    <td>antecedents</td>
    <td>consequents</td>
    <td>antecedent support</td>
    <td>consequent support</td>
    <td>support</td>
    <td>confidence</td>
    <td>lift</td>
    <td>...</td>
</tr>
<tr>
    <td>0</td>
    <td>(AVOCADO)</td>
    <td>(BREAD)</td>
    <td>0.12</td>
    <td>0.44</td>
    <td>0.08</td>
    <td>0.666667</td>
    <td>1.515152</td>
    <td>...</td>
</tr>
<tr>
    <td>1</td>
    <td>(BREAD)</td>
    <td>(AVOCADO)</td>
    <td>0.44</td>
    <td>0.12</td>
    <td>0.08</td>
    <td>0.181818</td>
    <td>1.515152</td>
    <td>...</td>
</tr>
</table>

```py
# Confidence
rules2 = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
rules2["antecedents_length"] = rules2["antecedents"].apply(lambda x: len(x))
rules2["consequents_length"] = rules2["consequents"].apply(lambda x: len(x))
rules2.sort_values("confidence")
```

```py
rules2.head()
```

<table>
<tr>
    <td></td>
    <td>antecedents</td>
    <td>consequents</td>
    <td>antecedent support</td>
    <td>consequent support</td>
    <td>support</td>
    <td>confidence</td>
    <td>lift</td>
    <td>...</td>
</tr>
<tr>
    <td>0</td>
    <td>(AVOCADO)</td>
    <td>(BREAD)</td>
    <td>0.12</td>
    <td>0.44</td>
    <td>0.08</td>
    <td>0.666667</td>
    <td>1.515152</td>
    <td>...</td>
</tr>
<tr>
    <td>1</td>
    <td>(BISCUIT)</td>
    <td>(BREAD)</td>
    <td>0.22</td>
    <td>0.44</td>
    <td>0.08</td>
    <td>0.363636</td>
    <td>0.826446</td>
    <td>...</td>
</tr>
</table>

Filter the dataframe by using standard pandas code. We are looking for a large lift (>=2) & confidence (>=0.6):

```py
print(rules2.loc[(rules2['lift']>=2) & (rules2['confidence']>= 0.6)] )
```

<table>
<tr>
    <td></td>
    <td>antecedents</td>
    <td>consequents</td>
    <td>antecedent support</td>
    <td>consequent support</td>
    <td>support</td>
    <td>confidence</td>
    <td>lift</td>
    <td>...</td>
</tr>
<tr>
    <td>3</td>
    <td>(CHEESE)</td>
    <td>(BREAD)</td>
    <td>0.10</td>
    <td>0.44</td>
    <td>0.01</td>
    <td>1.000000</td>
    <td>2.272727</td>
    <td>...</td>
</tr>
<tr>
    <td>10</td>
    <td>(SUGAR)</td>
    <td>(COFFEE)</td>
    <td>0.14</td>
    <td>0.24</td>
    <td>0.01</td>
    <td>0.714286</td>
    <td>2.976190</td>
    <td>...</td>
</tr>
</table>


# Visualizing results


1.Support vs Confidence

```py 
plt.scatter(rules2['support'], rules2['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()
```

2. Support vs Lift

```py 
plt.scatter(rules2['support'], rules2['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.show()
```



