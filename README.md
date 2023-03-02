# MarketBasketAnalysis

Task: show which products are frequently bought together.

## Importing the required libraries

```py
#for basic operations
import numpy as np
import pandas as pd

# for visualizations
import matplotlib.pyplot as plt

# for market basket analysis
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth, fpmax, fpcommon 
```

## Importing the dataset

```py
#read the data
store=pd.read_csv('~/MarketBasketAnalysis/GroceryStoreDataSet.csv',names=['product'],header=None)

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
    <th>product</th>
<tr>
    <td>MILK,BREAD,BISCUIT</td>
</tr>
<tr>
    <td>PIZZA</td>
</tr>
<tr>
    <td>PIZZA,WATER,BEER</td>
</tr>
</table>