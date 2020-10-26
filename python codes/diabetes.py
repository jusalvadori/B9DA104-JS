import csv
import pandas
import matplotlib.pyplot as plt
from collections import Counter

diabetes = open('C:/Temp/input/diabetes_data_upload.csv')
contents = csv.reader(diabetes)

for row in contents:
   print (row)


# read only the first line of the file
attributes = pandas.read_csv('C:/Temp/input/diabetes_data_upload.csv', nrows=1).columns.tolist()
print(attributes)
print(len(attributes))

# read dataset into Dataframe
df = pandas.read_csv('C:/Temp/input/diabetes_data_upload.csv')
print(df.head(2))
print(df['Age'].dtypes)
print(df['Gender'].dtypes)

#get column names in Pandas dataframe
# iterating the columns 
print('Attributes:')
for col in df.columns: 
    print(col) 

# list(data)
list(df.columns) 

#number of rows (instances)
df.count()
print('Rows:', len(df))

# Age Histogram
df['Age'].plot.hist()
df.mean(skipna = True)

# Gender pie chart
# https://towardsdatascience.com/pie-charts-in-python-302de204966c
print(Counter(df['Gender']))

gender_type = df.groupby('Gender').agg('count')
print(gender_type)

from matplotlib.gridspec import GridSpec
import numpy as np

type_labels = gender_type.Age.sort_values().index 
type_counts = gender_type.Age.sort_values()
print(type_labels)
print(type_counts)

plt.figure(1, figsize=(20,10)) 
the_grid = GridSpec(2, 2)

cmap = plt.get_cmap('coolwarm')
colors = [cmap(i) for i in np.linspace(0, 1, 8)]
explode = (0.1, 0 )  # explode 1st slice

#plt.subplot(the_grid[0, 1], aspect=1, title='Gender')
type_gender = plt.pie(type_counts, explode = explode, labels=type_labels, autopct='%1.1f%%', shadow=True, colors=colors)
plt.show()


# Class pie chart
class_type = df.groupby('class').agg('count')
print(class_type)

type_labels = class_type.Gender.sort_values().index 
type_counts = class_type.Gender.sort_values()
print(type_labels)
print(type_counts)

plt.figure(1, figsize=(20,10)) 
the_grid = GridSpec(2, 2)

cmap = plt.get_cmap('coolwarm')
colors = [cmap(i) for i in np.linspace(0, 1, 8)]
explode = (0.1, 0 )  # explode 1st slice

type_gender = plt.pie(type_counts, explode = explode, labels=type_labels, autopct='%1.1f%%', shadow=True, colors=colors)
plt.show()



















