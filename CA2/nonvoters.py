# -*- coding: utf-8 -*-
"""
Dublin Business Scholl
@author: Juliana Salvadori
@Student_number: 10521647
@Assigment: CA2 - Classification model

"""
#pip install missingno

import pandas as pd
from matplotlib import pyplot
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import missingno as mno
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns


'''
Load data
######################################
'''
file_name='nonvoters_data.csv'
data = pd.read_csv(file_name)

'''
Quick Glance
######################################
'''
view = data.head(10)
print(view)

# There are some columns that can be removed as they are not relevant for 
# classification process (RespID and weight) 
# Also, there some columns with too many NA values, can those columns be removed?

'''
Review Dimensions of the data
######################################
'''
dim = data.shape
print(dim)

# dataset has 5836 rows and 119 columns (118 features and 1 label)
# so, for that amount of rows we should use at maximum 6 features to
# apply machine learning algorithm
# that means, we can remove those columns with too many NA values
# as they probably will not be relevant for classification

'''
Clean data
######################################
'''
# Remove first (RespID) and second (weight) columns 
data = data.drop(['RespId', 'weight'], axis=1)

# Remove other columns where there are lots of missing data (NA)
data = data.drop(['Q22','Q29_1', 'Q29_2','Q29_3','Q29_4','Q29_5','Q29_6','Q29_7','Q29_8','Q29_9','Q29_10','Q31','Q32','Q33'], axis=1)

# Column Q1 has always value as 1 and 
# so it will not impact classification
data = data.drop(['Q1'], axis=1)

# Remove other columns where most of the answers are -1 (NA)
data = data.drop(['Q19_1','Q19_2','Q19_3','Q19_4','Q19_5','Q19_6','Q19_7','Q19_8','Q19_9','Q19_10',], axis=1)
data = data.drop(['Q28_1','Q28_2','Q28_3','Q28_4','Q28_5','Q28_6','Q28_7','Q28_8'], axis=1)

view = data.head(10)
print(view)

'''
Review data types
######################################
'''
types = data.dtypes
print(types)

# There are a few columns which the data type is object (string),
# and therefore, they need to be converted to integer values
# educ               object
# race               object
# gender             object
# income_cat         object
# voter_category     object

############################################
# Highest educational attainment category
educ_count = data.groupby('educ').size()
print (educ_count)

# There are 3 options for education
# High school or less    => 1
# Some college           => 2
# College                => 3

def set_educ_int (row):
   if row['educ'] == 'High school or less' :
      return 1
   if row['educ'] == 'Some college':
      return 2
   if row['educ'] == 'College' :
      return 3   
   return 0

data["educ_int"] = data.apply (lambda row: set_educ_int(row), axis=1)
view = data.loc[ :, ['educ', 'educ_int']]
print(view.head(10))
# Remove educ column
data = data.drop(['educ'], axis=1)

############################################
# Race of respondent
race_count = data.groupby('race').size()
print (race_count)

# There are 4 options for race
# Black           => 1
# Hispanic        => 2
# White           => 3
# Other/Mixed     => 4

def set_race_int (row):
   if row['race'] == 'Black' :
      return 1
   if row['race'] == 'Hispanic':
      return 2
   if row['race'] == 'White' :
      return 3   
   if row['race'] == 'Other/Mixed' :
      return 4   
   return 0

data["race_int"] = data.apply (lambda row: set_race_int(row), axis=1)
view = data.loc[ :, ['race', 'race_int']]
print(view.head(10))
# Remove race column
data = data.drop(['race'], axis=1)

############################################
# Gender of respondent
gender_count = data.groupby('gender').size()
print (gender_count)

# There are 2 options for gender
# Female    => 1
# Male      => 2

def set_gender_int (row):
   if row['gender'] == 'Female' :
      return 1
   if row['gender'] == 'Male':
      return 2   
   return 0

data["gender_int"] = data.apply (lambda row: set_gender_int(row), axis=1)
view = data.loc[ :, ['gender', 'gender_int']]
print(view.head(10))
# Remove gender column
data = data.drop(['gender'], axis=1)

############################################
# Household income category of respondent
income_count = data.groupby('income_cat').size()
print (income_count)

# There are 4 options for gender
# Less than $40k    => 1
# $40-75k           => 2
# $75-125k          => 3
# $125k or more     => 4

def set_income_int (row):
   if row['income_cat'] == 'Less than $40k' :
      return 1
   if row['income_cat'] == '$40-75k':
      return 2   
   if row['income_cat'] == '$75-125k':
      return 3   
   if row['income_cat'] == '$125k or more':
      return 4   
   return 0

data["income_int"] = data.apply (lambda row: set_income_int(row), axis=1)
view = data.loc[ :, ['income_cat', 'income_int']]
print(view.head(10))
# Remove income_cat column
data = data.drop(['income_cat'], axis=1)

############################################
# Voter category
voter_cat_count = data.groupby('voter_category').size()
print (voter_cat_count)

# There are 3 options for gender
# rarely/never    => 1
# sporadic        => 2
# always          => 3

def set_voter_cat_int (row):
   if row['voter_category'] == 'rarely/never' :
      return 1
   if row['voter_category'] == 'sporadic':
      return 2   
   if row['voter_category'] == 'always':
      return 3   
   return 0

data["voter_cat_int"] = data.apply (lambda row: set_voter_cat_int(row), axis=1)
view = data.loc[ :, ['voter_category', 'voter_cat_int']]
print(view.head(10))
# Remove voter_category column
data = data.drop(['voter_category'], axis=1)

types = data.dtypes
print(types)

view = data.head(10)
print(view)

'''
Data Description
######################################
'''
# limiting the results upto two possible digits after decimals 
pd.set_option('precision', 2)

# lists the number of non-null values and the datatype of each variable.
data.info()

# it shows that at this stage there is no NA values as all the
# features are listed as 5836 non-null rows 

# summary of statistics for the given dataset
description = data.describe()
print(description)

# this summary shows that even that there is no NA values
# some of the features have min value as -1, which is 
# not a valid and it will probably require 
# data imputation to replace those values


'''
Class distribution
######################################
'''
voter_cat_count = data.groupby('voter_cat_int').size()
print (voter_cat_count)

# voter_cat_int
# 1    1451
# 2    2574
# 3    1811

'''
Data Skew
######################################
'''
skew_data = data.skew()
print(skew_data.head(40))
print(skew_data.tail(45))

# When the value of the skewness is negative, the tail of the 
# distribution is longer towards the left hand side of the curve. 
# When the value of the skewness is positive, the tail of the 
# distribution is longer towards the right hand side of the curve.

#If the skewness is between -0.5 and 0.5, the data are fairly symmetrical.
#If the skewness is between -1 and – 0.5 or between 0.5 and 1, the data are moderately skewed.
#If the skewness is less than -1 or greater than 1, the data are highly skewed.

'''
Histogram to visually verify the skew
for each feature
(Univariate plot)
######################################
'''
fig = pyplot.figure(figsize = (20,20))
ax = fig.add_subplot(111)
data.hist(ax = ax)

'''
Density plot
(Univariate plot)
######################################
'''
K = data.plot(kind='density', figsize=(20,30),subplots=True,layout=(25,5),sharex=False,sharey=False)

# based on data skewness and histogram/density plot results to reduce the 
# number of features
data1 = data.loc[ :, ['educ_int','gender_int','income_int','ppage','Q11_3','Q14','Q15','Q16','Q17_1','Q17_2','Q17_3','Q17_4','Q2_10','Q2_2'	,'Q2_3'	,'Q2_4'	,'Q2_6'	,'Q2_8'	,'Q2_9'	,'Q23','Q24','Q25','Q27_1','Q27_2','Q27_3','Q27_4','Q27_5','Q27_6','Q3_1'	,'Q3_2'	,'Q3_3'	,'Q3_4'	,'Q3_5'	,'Q3_6'	,'Q30','Q4_1'	,'Q4_2'	,'Q4_3'	,'Q4_4'	,'Q4_5'	,'Q4_6'	,'Q5','Q6','Q7','Q8_1'	,'Q8_2'	,'Q8_3'	,'Q8_4'	,'Q8_5'	,'Q8_6'	,'Q8_7'	,'Q8_8'	,'Q8_9'	,'Q9_1'	,'Q9_2'	,'race_int', 'voter_cat_int']]

# review the density plot for the selected features
K = data1.plot(kind='density', figsize=(20,30),subplots=True,layout=(25,5),sharex=False,sharey=False)


# New glance to the data
view = data1.head(10)
print(view)
print(data1.shape)
# 56 features + 1 label

######################################
# let's have a look at the data description 
# for the remaining features
'''
Data Description review
######################################
'''
# set options to display all columns of the dataset
pd.options.display.max_columns = data1.shape[1]

# summary of statistics for the given dataset
description = data1.describe()
print(description)

# the summary shows that there are still features where min value is -1, 
# which is not a valid and it will probably require 
# data imputation to replace them or remove the rows with those invalid values
# before being able to use correlation and feature selection

# which columns have min as -1?
## extract min row as series
row = description.iloc[3]           
## filter for values equal to -11 and get columns via index
invalidMin = row[row == -1].index  
print(invalidMin)

# let's replace those -1 values for NA
for i in invalidMin:
    #print('index {}'.format(i))
    data1.loc[data1[i] == -1, i] = np.NAN          

# let's have a look at data
view = data1.head(10)
print(view)
 
# check how many NaN values for each feature
data1.isnull().sum()
data1.info()

# let's visualize the missing values
mno.matrix(data1, figsize = (20, 6))

# let's check the impact on the dataset size if 
# the lines with NA values are removed
print(len(data1))     
print(len(data1.dropna()))

# actual size = 5836
# removing NA values = 5019
# the difference is not significant, we will still have enough data
# so, instead of imput the missing values let's remove those lines

data1 = data1.dropna()
# check how many NaN values for each feature
data1.isnull().sum()


'''
Correlations matrix between Attributes
(Multivariate plot)
######################################
Correlation ranges from -1 to +1. Values closer to zero means there is no linear 
trend between the two variables. The close to 1 the correlation is the more 
positively correlated they are; that is as one increases so does the other and 
the closer to 1 the stronger this relationship is. A correlation closer to -1 
is similar, but instead of both increasing one variable will decrease as the 
other increases. The diagonals are all 1/dark collor because those squares are 
correlating each variable to itself (so it's a perfect correlation).
'''
correlations = data1.corr()
print(correlations)

col_names = list(data1)
# plot correlation matrix
fig = pyplot.figure(figsize=(10,10), dpi=100)
# 111: 1x1 grid, first subplot
ax = fig.add_subplot(111)
# normalize data using vmin, vmax
cax = ax.matshow(correlations, vmin=-1, vmax=1)
# add a colorbar to a plot.
fig.colorbar(cax)
# force matplotlib to use enough xticks so that all labels can be shown
ax.set_xticks(np.arange(len(col_names)))
ax.set_yticks(np.arange(len(col_names)))
# set x and y tick marks
ax.set_xticklabels(col_names, rotation=45)
ax.set_yticklabels(col_names)
# draw a matrix using the correlations data
pyplot.show()


'''
Data normalization
######################################
'''
# As all the features are categorical there is no need for normalization 


dim = data1.shape
print(dim)
# let's have a look at data
view = data1.head(10)
print(view)

'''
Feature Extraction with Univariate Statistical Tests 
(Chi-squared for classification)
######################################
'''
array = data1.values
X = array[:,0:56]
Y = array[:,56]
# feature extraction
test = SelectKBest(score_func = chi2, k = 3)
fit = test.fit(X, Y)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
score = fit.scores_.tolist()

col_names = list(data1)

for i in col_names:
    print(i)
for i in score:
    print(i)
    
# let's use the top 5 features with the highest scores
#	ppage	2930.821374
#	Q27_4	236.0515466
#	Q27_3	221.3262291
#	Q27_2	218.0110906
#	Q27_6	217.2500834
    
#	Q27_1	213.9802858
#	Q27_5	210.5921937
#	Q25	205.4040102
#	Q30	146.8340542
#	Q2_2	128.8242681
#	Q2_3	100.020342
#	Q16	89.6873252
#	Q9_1	71.5800691
#	income_int	67.93467606
#	Q17_2	60.76622649
#	Q17_1	52.8789998
#	Q4_2	51.61718718
#	Q4_1	43.68697129
#	Q8_8	40.22266807
#	Q6	38.4131555
#	educ_int	37.05459512
#	Q4_3	33.54947249
#	Q8_5	31.97564284
#	Q23	31.15760338
#	Q2_6	29.93346927
#	Q3_4	29.35093514
#	Q4_5	28.10775865
#	Q5	27.0635527
#	Q17_3	23.60226392
#	Q8_6	20.09420049
#	Q9_2	20.05601247
#	Q8_3	19.59373171
#	Q4_4	15.89421994
#	Q4_6	14.21729696
#	Q3_5	10.37095451
#	Q3_3	10.18461851
#	Q2_10	9.719726731
#	Q7	9.102052055
#	Q2_4	8.118405815
#	Q8_9	7.982289167
#	Q8_7	7.971945716
#	Q3_2	7.062411801
#	Q8_4	7.022067141
#	Q2_9	6.71312638
#	Q14	6.075426242
#	Q2_8	4.489968185
#	Q8_1	3.235156923
#	Q15	2.610157283
#	Q3_6	1.827751882
#	Q3_1	1.716918416
#	gender_int	1.503223626
#	Q8_2	1.061391986
#	Q17_4	0.970515833
#	Q24	0.767749812
#	race_int	0.454725598
#	Q11_3	0.180395344
    

'''
K Nearest Neighbor Algorithm
for classification
######################################
'''
# select features
x = data1[['ppage','Q27_4','Q27_3','Q27_2','Q27_6']]
# select target  (actual values)
y = data1[['voter_cat_int']]
# split the dataset in test (25%) and train (75%)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
# apply KNN algorithm and looks for the 5 nearst neighbors using Euclidean distance
knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_classifier.fit(x_train, y_train)
# use the model to predict the voter category for test dataset
y_pred = knn_classifier.predict(x_test)
#print(x_test)
#print(y_test)  # actual test values

df = pd.DataFrame(y_pred, columns=["voter_cat_int"]) 
pred = df.groupby('voter_cat_int').size()
print (pred)
# voter_cat_int count
# 1             317     rarely/never
# 2             517     sporadic
# 3             421     always

######################################
# let's evaluate the model using confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#A      Predict
#c  [[195  65  37]
#t   [ 89 309 159]
#u   [ 33 143 225]]
#a
#l

'''
Confusion matrix visualization
######################################
'''
index = ['rarely/never', 'sporadic', 'always']  
columns = ['rarely/never', 'sporadic', 'always']  
cm_df = pd.DataFrame(cm,columns,index)                      
pyplot.figure(figsize=(15,10))  
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues') #annot=True to annotate cells

# accuracy of the model
diagonal_sum = cm.trace()
sum_of_all_elements = cm.sum()
accuracy = (diagonal_sum / sum_of_all_elements) *100;
print('Model Accuracy is',accuracy)


















