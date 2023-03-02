# MarketBasketAnalysis

Task: show which products are frequently bought together.

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

<p>(50,1)</p>

```py
#Statistical description of the dataset.
print(store.describe())
```
<table>
<tr>
    <th>product count</th>
    <th>unique</th>
    <th>top</th>
    <th>freq</th>
</tr>
<tr>
    <td>50</td>
    <td>47</td>           
    <td>BREAD,COFFEE,SUGAR</td>
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
...
</table>