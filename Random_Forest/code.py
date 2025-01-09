# Training a Random Forest
# While tuning the hyperparameters of a single decision tree may lead to some improvements, a much more effective strategy is to combine the results of several decision trees trained with slightly different parameters. This is called a random forest model.

# The key idea here is that each decision tree in the forest will make different kinds of errors, and upon averaging, many of their errors will cancel out. This idea is also commonly known as the "wisdom of the crowd":

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

plt.title('No. of Rows per Year')
sns.countplot(x=pd.to_datetime(raw_df.Date).dt.year)
plt.show()

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

numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'mean').fit(raw_df[numeric_cols])
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(raw_df[numeric_cols])
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(raw_df[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]
print(X_test)


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs=-1, random_state=42)

model.fit(X_train, train_targets)

model.score(X_train, train_targets)

model.score(X_val, val_targets)

train_probs = model.predict_proba(X_train)
train_probs
# We can can access individual decision trees using model.estimators_
model.estimators_[0]

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Let's create a base model with which we can compare models with tuned hyperparameters.
base_model = RandomForestClassifier(random_state=42, n_jobs=-1).fit(X_train, train_targets)
base_train_acc = base_model.score(X_train, train_targets)
base_val_acc = base_model.score(X_val, val_targets)

base_accs = base_train_acc, base_val_acc
base_accs

### `n_estimators`

# This argument controls the number of decision trees in the random forest. The default value is 100. For larger datasets, it helps to have a greater number of estimators. As a general rule, try to have as few estimators as needed.


# **10 estimators**

model = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=10)
model.fit(X_train, train_targets)
model.score(X_train, train_targets), model.score(X_val, val_targets)

model = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=500)
model.fit(X_train, train_targets)
model.score(X_train, train_targets)

def test_params(**params):
    model = RandomForestClassifier(random_state=42, n_jobs=-1, **params).fit(X_train, train_targets)
    return model.score(X_train, train_targets), model.score(X_val, val_targets)

test_params(max_depth=26)

### `max_features`

# Instead of picking all features (columns) for every split, we can specify that only a fraction of features be chosen randomly to figure out a split.


# Notice that the default value `auto` causes only $\sqrt{n}$ out of total features ( $n$ ) to be chosen
# randomly at each split. This is the reason each decision tree in the forest is different. While it may
# seem counterintuitive, choosing all features for every split of every tree will lead to identical trees, so the random forest will not generalize well.

test_params(max_features='log2')


### `min_samples_split` and `min_samples_leaf`

# By default, the decision tree classifier tries to split every node that has 2 or more.
# You can increase the values of these arguments to change this behavior and reduce overfitting,
# especially for very large datasets.
test_params(min_samples_split=3, min_samples_leaf=2)

# ### `min_impurity_decrease`

# This argument is used to control the threshold for splitting nodes.
# A node will be split if this split induces a decrease of the impurity (Gini index) greater than or
# equal to this value. It's default value is 0, and you can increase it to reduce overfitting.

test_params(min_impurity_decrease=1e-7)

### `bootstrap`, `max_samples`

# By default, a random forest doesn't use the entire dataset for training each decision tree.
# Instead it applies a technique called bootstrapping. For each tree, rows from the dataset are
# picked one by one randomly, with replacement i.e. some rows may not show up at all, while some rows may show up multiple times.

# Bootstrapping helps the random forest generalize better, because each decision tree only sees a
# fraction of th training set, and some rows randomly get higher weightage than others.

test_params(class_weight='balanced')
test_params(class_weight={'No': 1, 'Yes': 2})

model = RandomForestClassifier(n_jobs=-1,
                               random_state=42,
                               n_estimators=500,
                               max_features=7,
                               max_depth=30,
                               class_weight={'No': 1, 'Yes': 1.5})


def predict_input(model, single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob

import joblib

aussie_rain = {
    'model': model,
    'imputer': imputer,
    'scaler': scaler,
    'encoder': encoder,
    'input_cols': input_cols,
    'target_col': target_col,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'encoded_cols': encoded_cols
}
joblib.dump(aussie_rain, 'aussie_rain.joblib')
aussie_rain2 = joblib.load('aussie_rain.joblib')

test_preds2 = aussie_rain2['model'].predict(X_test)
accuracy_score(test_targets, test_preds2)