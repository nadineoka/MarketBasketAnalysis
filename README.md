# MarketBasketAnalysis

Task: show which products are frequently bought together.

```
#import libraries
import numpy as np
import pandas as pd

#read the data
store=pd.read_csv('~/MarketBasketAnalysis/GroceryStoreDataSet.csv',names=['product'],header=None)
```

```
check data set
print(store.shape) 
```

(50,1)

```
#Statistical description of the dataset.
print(store.describe())
```

product count           50
unique                  47
top     BREAD,COFFEE,SUGAR
freq                     2

 
