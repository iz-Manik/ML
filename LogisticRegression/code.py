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