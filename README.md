# MarketBasketAnalysis

Task: show which products are frequently bought together.

## Importing the required libraries.

```py
#import libraries
import numpy as np
import pandas as pd

#read the data
store=pd.read_csv('~/MarketBasketAnalysis/GroceryStoreDataSet.csv',names=['product'],header=None)
```

```py
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
    <td>MILK,BREAD,BISCUIT</td>
</tr>
<tr>
    <td>PIZZA</td>
</tr>
<tr>
    <td>PIZZA,WATER,BEER</td>
</tr>
<tr>
    <td>WATER,AVOCADO,BREAD</td>
</tr>

</table>
...