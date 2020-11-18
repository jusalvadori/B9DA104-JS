# -*- coding: utf-8 -*-
"""
Dublin Business School
@author: Juliana Salvadori
@Student_number: 10521647
@Assigment: CA2 - Regression model

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot  
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, r2_score

'''
Load data
######################################
'''
file_name='movies.csv'
data = pd.read_csv(file_name)
# set options to display all columns of the dataset
pd.options.display.max_columns = data.shape[1]

'''
Quick Glance
######################################
'''
view = data.head(10)
print(view)

# The data looks, it does not show too many missing values
# and the continuous are in the same scale


'''
Review Dimensions of the data
######################################
'''
dim = data.shape
print(dim)
# dataset has 1794 rows and 15 columns (14 features and 1 target)

'''
Clean data
######################################
'''
# let's check how many NaN values for each feature
data.isnull().sum()
# let's check the impact on the dataset size if 
# the lines with NA values are removed
print(len(data.dropna()))

# the dataset will be reduced to 1600 rows, which is fine
# so, there is no need to keep those values and work with
# imputation methods
data = data.dropna()
# check that none NaN values was left
data.isnull().sum()

dim = data.shape
print(dim)
# 1600 rows 

'''
Review data types
######################################
'''
types = data.dtypes
print(types)

# Note that the data type for one of the features that will be used (binary) is object (string)
# So, I will need to convert that to integer values

'''
Class distribution
######################################
'''
# binary
binary_count = data.groupby('binary').size()
print (binary_count)

# There are 2 options for binary
# FAIL    856 => 0
# PASS    744 => 1

def set_binary_int (row):
   if row['binary'] == 'PASS':
      return 1
   return 0

data["binary_int"]  = data.apply (lambda row: set_binary_int(row), axis=1)

view = data.loc[ :, ['binary', 'binary_int']]
print(view.head(10))

binary_count = data.groupby('binary_int').size()
print (binary_count)


'''
Data Description
######################################
'''
# limiting the results up to two possible digits after decimals 
pd.set_option('precision', 2)

# lists the number of non-null values and the datatype of each variable.
data.info()

# it shows that at this stage there is no NaN values 

# summary of statistics for the given dataset
description = data.describe()
print(description)

# it shows that the target variable (budget_2013$) and features (domgross_2013$,intgross_2013$,binary_int)
# have no negative or NaN values

'''
Correlations between Target and Feature
######################################
'''
columns = ['binary_int','budget_2013$','domgross_2013$','intgross_2013$']
correlations = data[columns].corr()
print(correlations)


'''
Rescaling data 
######################################
'''
x_feature = ['binary_int','domgross_2013$', 'intgross_2013$']
y_target  = ['budget_2013$']

x = data[x_feature].values
scale = MinMaxScaler (feature_range = (0,2))
rescaled_data = scale.fit_transform(x)
np.set_printoptions(precision=3)
print(rescaled_data[0:5,:])
print(rescaled_data.shape)

x = rescaled_data
y = data[y_target].values


'''
Applying Linear regression
######################################
'''
# Splitting the dataset into training and test set
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)  

print(len(x_test))   # 320  (20%)
print(len(y_test))
print(len(x_train))  # 1280 (80%)
print(len(y_train))

# Fitting the model to the training set
model = LinearRegression()  
model.fit(x_train, y_train)  

# Checking the performance of the model by predicting the test set result
y_pred = model.predict(x_test)  
print(len(y_pred))

print('Train Score: ', model.score(x_train, y_train))  # 0.5141395306771033
print('Test Score: ', model.score(x_test, y_test))     # 0.40797528086306056


# Note that for RSME, the lower that value is, the better the fit
# The ideal MSE isn't 0, since then you would have a model that 
# perfectly predicts your training data, but which is very unlikely 
# to perfectly predict any other data. What you want is a balance 
# between overfit (very low MSE for training data) and 
# underfit (very high MSE for test/validation/unseen data)
test_set_rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
# R2: the closer towards 1, the better the fit
test_set_r2   = r2_score(y_test, y_pred)

print('Mean squared error:',test_set_rmse)
print('Variance score:',test_set_r2)

# RSME: 41063032.86551617
# R2: 0.40797528086306045

######################################
# Density plot of test Actual data and test Predict data
#
compare = pd.DataFrame(data= y_test, columns =['Actual'])
compare['Predicted'] = y_pred

fig, ax = pyplot.subplots(1,1)
compare['Actual'].plot(kind='density')
compare['Predicted'].plot(kind='density')
pyplot.legend()
fig.show()

'''
KFolds cross-validation to evaluate the model
######################################
'''
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor 
from sklearn.metrics import mean_absolute_error

kf = KFold (n_splits=20)
list_training_error = []
list_testing_error = []
for train_idx, test_idx in kf.split(x):
    # 
    x_trainr, x_testr = x[train_idx], x[test_idx]
    y_trainr, y_testr = y[train_idx], y[test_idx]
    
    x_train = np.array(x_trainr[x_feature]).reshape(-1,1)
    y_train = np.array(y_trainr[y_target]).reshape(-1,1)
    x_test = np.array(x_testr[x_feature]).reshape(-1,1)
    y_test = np.array(y_testr[y_target]).reshape(-1,1)    
    
    # fit the model
    model = MLPRegressor()
    model.fit(x_train, y_train)
    
    # prediction
    y_train_pred = model.predict(x_train)
    y_test_pred  = model.predict(x_test)
    
    # estimate the errors for test and training sets
    fold_training_error = mean_absolute_error(y_train, y_train_pred) 
    fold_testing_error = mean_absolute_error(y_test, y_test_pred)
    
    list_training_error.append(fold_training_error)
    list_testing_error.append(fold_testing_error)
    
print(type(y_train_pred))    
    
    
pyplot.style.use('ggplot')    
pyplot.figure( figsize=(15,10))
pyplot.subplot(1,2,1)
pyplot.plot(range(1, kf.get_n_splits() + 1), np.array(list_training_error).ravel(), 'o-')
pyplot.xlabel('number of fold')
pyplot.ylabel('training error')
pyplot.title('Training error across folds')
pyplot.tight_layout()
pyplot.subplot(1,2,2)
pyplot.plot(range(1, kf.get_n_splits() + 1), np.array(list_testing_error).ravel(), 'o-')
pyplot.xlabel('number of fold')
pyplot.ylabel('testing error')
pyplot.title('Testing error across folds')
pyplot.tight_layout()
pyplot.show()
