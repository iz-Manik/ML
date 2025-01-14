medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'
from urllib.request import urlretrieve
urlretrieve(medical_charges_url, 'medical.csv')
import pandas as pd
medical_df = pd.read_csv('medical.csv')
print(medical_df)
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
fig = px.histogram(medical_df,
                   x='age',
                   marginal='box',
                   nbins=47,
                   title='Distribution of Age')
fig.update_layout(bargap=0.1)
fig.show()
fig = px.histogram(medical_df,
                   x='bmi',
                   marginal='box',
                   color_discrete_sequence=['red'],
                   title='Distribution of BMI (Body Mass Index)')
fig.update_layout(bargap=0.1)
fig.show()
fig = px.histogram(medical_df,
                   x='charges',
                   marginal='box',
                   color='smoker',
                   color_discrete_sequence=['green', 'grey'],
                   title='Annual Medical Charges')
fig.update_layout(bargap=0.1)
fig.show()
px.histogram(medical_df, x='smoker', color='sex', title='Smoker')
fig = px.scatter(medical_df,
                 x='age',
                 y='charges',
                 color='smoker',
                 opacity=0.8,
                 hover_data=['sex'],
                 title='Age vs. Charges')
fig.update_traces(marker_size=5)
fig.show()

#Correlation
medical_df.charges.corr(medical_df.age)

#To compute the correlation for categorical columns, they must first be converted into numeric columns.
smoker_values = {'no': 0, 'yes': 1}
smoker_numeric = medical_df.smoker.map(smoker_values)
medical_df.charges.corr(smoker_numeric)

numeric_df = medical_df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), cmap='Reds', annot=True)
plt.title('Correlation Matrix');
plt.show()


#Linear Regression using single feature
non_smoker_df = medical_df[medical_df.smoker == 'no']
plt.title('Age vs. Charges')
sns.scatterplot(data=non_smoker_df, x='age', y='charges', alpha=0.7, s=15);

#A line on the X&Y coordinates has the following formula:
#y=wx+b
#The numbers w and b are called the parameters or weights of the model.

def estimate_charges(age, w, b):
    return w * age + b

w = 50
b = 100

ages = non_smoker_df.age
estimated_charges = estimate_charges(ages, w, b)
target = non_smoker_df.charges

plt.plot(ages, estimated_charges, 'r', alpha=0.9);
plt.scatter(ages, target, s=8,alpha=0.8);
plt.xlabel('Age');
plt.ylabel('Charges')
plt.legend(['Estimate', 'Actual']);

import numpy as np
def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))

targets = non_smoker_df['charges']
predicted = estimate_charges(non_smoker_df.age, w, b)

rmse(targets, predicted)

#Linear Regression using Scikit-learn
from sklearn.linear_model import LinearRegression
model = LinearRegression()

#fit(X, y, sample_weight=None) method of sklearn.linear_model._base.LinearRegression instance
    #Fit linear model.

    #Parameters
    # ----------
    # X : {array-like, sparse matrix} of shape (n_samples, n_features)
    #     Training data

    # y : array-like of shape (n_samples,) or (n_samples, n_targets)
    #     Target values. Will be cast to X's dtype if necessary

inputs = non_smoker_df[['age']]
targets = non_smoker_df.charges
print('inputs.shape :', inputs.shape)
print('targes.shape :', targets.shape)

model.fit(inputs, targets)

model.predict(np.array([[23],
                        [37],
                        [61]]))

# w
model.coef_
# b
model.intercept_

predictions = model.predict(inputs)

#charges=w1*age+w2*bmi+b

# Create inputs and targets
inputs, targets = non_smoker_df[['age', 'bmi']], non_smoker_df['charges']

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)

sns.barplot(data=medical_df, x='smoker', y='charges');
smoker_codes = {'no': 0, 'yes': 1}
medical_df['smoker_code'] = medical_df.smoker.map(smoker_codes)

# We can now use the `smoker_df` column for linear regression.

# charges = w1 * age + w2 * bmi + w3 * children + w4 * smoker + b
# Create inputs and targets
inputs, targets = medical_df[['age', 'bmi', 'children', 'smoker_code']], medical_df['charges']

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)

from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit(medical_df[['region']])
enc.categories_

one_hot = enc.transform(medical_df[['region']]).toarray()
one_hot
# array([[0., 0., 0., 1.],
#        [0., 0., 1., 0.],
#        [0., 0., 1., 0.],
#        ...,
#        [0., 0., 1., 0.],
#        [0., 0., 0., 1.],
#        [0., 1., 0., 0.]])

# Because different columns have different ranges, we run into two issues:

# We can't compare the weights of different column to identify which features are important
# A column with a larger range of inputs may disproportionately affect the loss and dominate the optimization process.
# For this reason, it's common practice to scale (or standardize) the values in numeric column by subtracting the mean and dividing by the standard deviation.

from sklearn.preprocessing import StandardScaler
numeric_cols = ['age', 'bmi', 'children']
scaler = StandardScaler()
scaler.fit(medical_df[numeric_cols])

scaler.mean_

scaler.var_

scaled_inputs = scaler.transform(medical_df[numeric_cols])
scaled_inputs