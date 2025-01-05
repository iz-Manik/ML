# Logistic regression is a commonly used technique for solving binary classification problems. In a logistic regression model:

# we take linear combination (or weighted sum of the input features)
# we apply the sigmoid function to the result to obtain a number between 0 and 1
# this number represents the probability of the input being classified as "Yes"
# instead of RMSE, the cross entropy loss function is used to evaluate the results


import opendatasets as od

dataset_url = 'https://www.kaggle.com/jsphyg/weather-dataset-rattle-package'
#od.download(dataset_url)

import os

print("Current Working Directory:", os.getcwd())
data_dir = './logisticregression/weather-dataset-rattle-package'

# Check if the directory exists
if os.path.exists(data_dir):
    print(os.listdir(data_dir))
else:
    print(f"Directory {data_dir} does not exist.")


train_csv = data_dir + '/weatherAUS.csv'
import pandas as pd
raw_df = pd.read_csv(train_csv)
raw_df

import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

px.histogram(raw_df, x='Location', title='Location vs. Rainy Days', color='RainToday')

px.histogram(raw_df,
             x='Temp3pm',
             title='Temperature at 3 pm vs. Rain Tomorrow',
             color='RainTomorrow')\

px.scatter(raw_df.sample(2000),
           title='Temp (3 pm) vs. Humidity (3 pm)',
           x='Temp3pm',
           y='Humidity3pm',
           color='RainTomorrow')


from sklearn.model_selection import train_test_split

train_val_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

plt.title('No. of Rows per Year')
sns.countplot(x=pd.to_datetime(raw_df.Date).dt.year);

year = pd.to_datetime(raw_df.Date).dt.year

train_df = raw_df[year < 2015]
val_df = raw_df[year == 2015]
test_df = raw_df[year > 2015]

input_cols = list(train_df.columns)[1:-1]
target_col = 'RainTomorrow'

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()

val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()

test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

import numpy as np

numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()


### Imputing Missing Numeric Data

# Machine learning models can't work with missing numerical data. The process of filling missing values is called imputation.

# <img src="https://i.imgur.com/W7cfyOp.png" width="480">

# There are several techniques for imputation, but we'll use the most basic one: replacing missing values with the average value in the column using the `SimpleImputer` class from `sklearn.impute`.

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = 'mean')

imputer.fit(raw_df[numeric_cols])

train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

## Scaling Numeric Features

# Another good practice is to scale numeric features to a small range of values e.g. $(0,1)$ or $(-1,1)$. Scaling numeric features ensures that no particular feature has a disproportionate impact on the model's loss. Optimization algorithms also work better in practice with smaller numbers.

# The numeric columns in our dataset have varying ranges.

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(raw_df[numeric_cols])

train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])