# author: zhangda
# date:   2021/9/5 10:44
# note:   .


import pandas as pd
import numpy as np

data = pd.read_csv('criteo_sampled_data.csv')
data.head()

data.describe()
data.shape
data.info()
data['label'].value_counts()
data.info()