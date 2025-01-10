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

plt.figure(figsize=(18,8))
temp_df = merged_df.sample(40000)
sns.scatterplot(x=temp_df.Sales, y=temp_df.Customers, hue=temp_df.Date.dt.year, alpha=0.8)
plt.title("Sales Vs Customers")
plt.show()

plt.figure(figsize=(18,8))
temp_df = merged_df.sample(10000)
sns.scatterplot(x=temp_df.Store, y=temp_df.Sales, hue=temp_df.Date.dt.year, alpha=0.8)
plt.title("Stores Vs Sales")
plt.show()

merged_df.corr()['Sales'].sort_values(ascending=False)

### Feature Engineering

# Feature engineer is the process of creating new features (columns) by transforming/combining
# existing features or by incorporating data from external sources.


# For example, here are some features that can be extracted from the "Date" column:

# 1. Day of week
# 2. Day or month
# 3. Month
# 4. Year
# 5. Weekend/Weekday
# 6. Month/Quarter End

train_size = int(.75 * len(merged_df))
train_size

sorted_df = merged_df.sort_values('Date')
train_df, val_df = sorted_df[:train_size], sorted_df[train_size:]

input_cols = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'StoreType', 'Assortment', 'Day', 'Month', 'Year']
target_col = 'Sales'

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()

val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()

test_inputs = merged_test_df[input_cols].copy()
# Test data does not have targets
numeric_cols = ['Store', 'Day', 'Month', 'Year']
categorical_cols = ['DayOfWeek', 'Promo', 'StateHoliday', 'StoreType', 'Assortment']

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean').fit(train_inputs[numeric_cols])
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(train_inputs[numeric_cols])
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

def return_mean(inputs):
    return np.full(len(inputs), merged_df.Sales.mean())

train_preds = return_mean(X_train)

from sklearn.metrics import mean_squared_error, root_mean_squared_error
root_mean_squared_error(train_preds, train_targets, squared=False)
root_mean_squared_error(return_mean(X_val), val_targets, squared=False)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, train_targets)
train_preds = linreg.predict(X_train)
train_preds

root_mean_squared_error(train_preds, train_targets, squared=False)
# 2741.5918405767957

val_preds = linreg.predict(X_val)
val_preds
root_mean_squared_error(val_preds, val_targets, squared=False)
# 2817.5187414380916

def try_model(model):
    # Fit the model
    model.fit(X_train, train_targets)

    # Generate predictions
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    # Compute RMSE
    train_rmse = root_mean_squared_error(train_targets, train_preds, squared=False)
    val_rmse = root_mean_squared_error(val_targets, val_preds, squared=False)
    return train_rmse, val_rmse

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor

try_model(LinearRegression())
try_model(Ridge())
try_model(Lasso())
try_model(ElasticNet())
try_model(SGDRegressor())

from sklearn.tree import DecisionTreeRegressor, plot_tree
tree = DecisionTreeRegressor(random_state=42)
try_model(tree)
plt.figure(figsize=(40, 20))
plot_tree(tree, max_depth=3, filled=True, feature_names=numeric_cols+encoded_cols);

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
try_model(rf)