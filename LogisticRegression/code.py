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