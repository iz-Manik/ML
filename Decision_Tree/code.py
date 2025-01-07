import opendatasets as od
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

od.download('https://www.kaggle.com/jsphyg/weather-dataset-rattle-package')

raw_df = pd.read_csv('weather-dataset-rattle-package/weatherAUS.csv')

print(raw_df)

print(raw_df.info())

raw_df.dropna(subset=['RainTomorrow'], inplace=True)

## Preparing the Data for Training

# We'll perform the following steps to prepare the dataset for training:

# 1. Create a train/test/validation split
# 2. Identify input and target columns
# 3. Identify numeric and categorical columns
# 4. Impute (fill) missing numeric values
# 5. Scale numeric values to the $(0, 1)$ range
# 6. Encode categorical columns to one-hot vectors


