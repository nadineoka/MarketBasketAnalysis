# MarketBasketAnalysis

<style>
      html * {
        font-size: 16px;
        line-height: 1.625;
        color: #2020131;
        font-family: Nunito, sans-serif;
      }
      
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
    <td>product count</td>
    <td>unique</td>
    <td>top</td>
    <td>freq</td>
</tr>
<tr>
    <td>50</td>
    <td>47</td>           
    <td>BREAD,COFFEE,SUGAR</td>
    <td>2</td>
</tr>
</table>
 
