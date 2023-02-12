import numpy as np
import pandas as pd
import scipy as sp
import time as tm

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Load the csv file as data frame
df = pd.read_csv('weatherAUS.csv')
# print('Size of data frame is', df.shape)

# Display the data
# print(df[0:5])

# Data pre processing
# Checking null values
# print(df.count().sort_values())

# Drop unneccessary columns
df.drop(columns=['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am', 'Location', 'Date'], axis=1, inplace=True)
# print('After dropping columns', df.shape)

# Removing any null values
df.dropna(how='any', inplace=True)
# print('After removing null', df.shape)

# Removing any outliers
zscre = sp.stats.zscore(df._get_numeric_data())
z = np.abs(zscre)
# print('Outliers \n', z)

df = df[(z < 3).all(axis=1)]
# print('After removing outliers', df.shape)

# Changing yes/no to 0/1
df['RainToday'].replace({'Yes' : 1, 'No' : 0}, inplace=True)
df['RainTomorrow'].replace({'Yes' : 1, 'No' : 0}, inplace=True)

# See unique values and convert them to int
category = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
for col in category:
    # print(col, ':', np.unique(df[col]))
    pass

# Transform the category columns
df = pd.get_dummies(df, columns=category)
# print('Changing after int \n', df.iloc[4:9])

# Standardise the data
scaler = preprocessing.MinMaxScaler()
scaler.fit(df)
trnfm = scaler.transform(df)
df = pd.DataFrame(trnfm, index=df.index, columns=df.columns)
print('Standardise data \n', df.iloc[4:9])

# Feature Selection
# Exploratory data analysis
x = df.loc[:, df.columns != 'RainTomorrow']
y = df['RainTomorrow']

selector = SelectKBest(chi2, k=3)
selector.fit(x, y)
x_now = selector.transform(x)

x_t = x.columns[selector.get_support(indices=True)]
# print('Top 3 columns :', x_t)

# Choose one of the feature
# ['Rainfall', 'Humidity3pm', 'RainToday']
x = df[['Rainfall']]
y = df[['RainTomorrow']]

# Data Splicing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Data Modelling

# LogisticRegression
t0 = tm.time()
logreg = LogisticRegression(random_state=0)
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
score = accuracy_score(y_test, y_pred) * 100

print('Accuracy of LogisticRegression:', score)
print('Time of LogisticRegression:', tm.time() - t0)

# RandomForestClassifier
t0 = tm.time()
rfc = RandomForestClassifier(random_state=0)
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
score = accuracy_score(y_test, y_pred) * 100

print('Accuracy of RandomForestClassifier:', score)
print('Time of RandomForestClassifier:', tm.time() - t0)

# DecisionTreeClassifier
t0 = tm.time()
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)
score = accuracy_score(y_test, y_pred) * 100

print('Accuracy of DecisionTreeClassifier:', score)
print('Time of DecisionTreeClassifier:', tm.time() - t0)

# SupportVectorMachine
t0 = tm.time()
svc = SVC(kernel='linear')
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
score = accuracy_score(y_test, y_pred) * 100

print('Accuracy of SupportVectorMachine:', score)
print('Time of SupportVectorMachine:', tm.time() - t0)
