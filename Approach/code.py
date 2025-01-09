import os
import matplotlib
import opendatasets as od
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

# Loss Functions and Evaluation Metrics
# Once you have identified the type of problem you're solving, you need to pick an appropriate evaluation metric. Also, depending on the kind of model you train, your model will also use a loss/cost function to optimize during the training process.

# Evaluation metrics - they're used by humans to evaluate the ML model

# Loss functions - they're used by computers to optimize the ML model

### Downloading Data

# There may be different sources to get the data:

# * CSV files
# * SQL databases
# * Raw File URLs
# * Kaggle datasets
# * Google Drive
# * Dropbox
# * etc.

# Identify the right tool/library to get the data.

# For the Rossmann Store Sales prediction dataset, we'll use the `opendatasets` library.
# Make sure to [accept the competition rules](https://www.kaggle.com/c/rossmann-store-sales/rules) before
# executing the following cell.

od.download('https://www.kaggle.com/c/rossmann-store-sales')

ross_df = pd.read_csv('./rossmann-store-sales/train.csv', low_memory=False)
print(ross_df)

store_df = pd.read_csv('./rossmann-store-sales/store.csv')

merged_df = ross_df.merge(store_df, how='left', on='Store')
merged_df

test_df = pd.read_csv('rossmann-store-sales/test.csv')
merged_test_df = test_df.merge(store_df, how='left', on='Store')

merged_df = merged_df[merged_df.Open==1].copy()
sns.histplot(data=merged_df, x='Sales');