import numpy as np 
import pandas as pd
from sklearn import svm
from datetime import datetime, date

print("Reading in training data")
# raw_train = np.array(list(csv.reader(open("../data/train.csv", "rb"), delimiter=',')))
raw_train = pd.read_csv('../data/train.csv')

print("Reading in test data")
# raw_test = np.array(list(csv.reader(open("../data/test.csv", "rb"), delimiter=',')))
raw_test = pd.read_csv('../data/test.csv')

print("Removing unused columns")
raw_train.drop('Descript', axis=1, inplace=True)
raw_train.drop('PdDistrict', axis=1, inplace=True)
raw_train.drop('Resolution', axis=1, inplace=True)
raw_train.drop('Address', axis=1, inplace=True)
raw_test.drop('PdDistrict', axis=1, inplace=True)
raw_test.drop('Address', axis=1, inplace=True)

# Bin the GPS coordinates to ~100m radius precision
print("Binning GPS coordinates")
raw_train.X, raw_train.Y = raw_train.X.round(decimals=3), raw_train.X.round(decimals=3)
raw_test.X, raw_test.Y = raw_test.X.round(decimals=3), raw_test.X.round(decimals=3)

# Map day of week to integers from 1 - 7
print("Mapping day of the week")
days_of_week = list(set(raw_train.DayOfWeek))
days_of_week_dict = {days_of_week[x]: float(x) for x in range(len(days_of_week))}
raw_train.DayOfWeek = raw_train.DayOfWeek.map(days_of_week_dict)
raw_test.DayOfWeek = raw_test.DayOfWeek.map(days_of_week_dict)

print("Mapping categories")

categories = list(set(raw_train.Category))
categories_dict = {categories[x]: float(x) for x in range(len(categories))}
raw_train_category = raw_train.Category.map(categories_dict)

# Add columns to pandas for year, month, hour
print("Extracting year, month, hour to separate columns within training (long process)")
raw_train['Year'] = raw_train.Dates.map(lambda date: datetime.strptime(date, '%Y-%m-%d %H:%M:%S').year)
raw_train['Month'] = raw_train.Dates.map(lambda date: datetime.strptime(date, '%Y-%m-%d %H:%M:%S').month)
raw_train['Hour'] = raw_train.Dates.map(lambda date: datetime.strptime(date, '%Y-%m-%d %H:%M:%S').hour)

print("Extracting year, month, hour to separate columns within test (long process)")
raw_test['Year'] = raw_test.Dates.map(lambda date: datetime.strptime(date, '%Y-%m-%d %H:%M:%S').year)
raw_test['Month'] = raw_test.Dates.map(lambda date: datetime.strptime(date, '%Y-%m-%d %H:%M:%S').month)
raw_test['Hour'] = raw_test.Dates.map(lambda date: datetime.strptime(date, '%Y-%m-%d %H:%M:%S').hour)

# Delete unnecessary data columns, training should match test columns
print("Deleting unnecessary columns out of training")
raw_train.drop('Dates', axis=1, inplace=True)
raw_train.drop('Category', axis=1, inplace=True)

# extract test id column
print("Extracting Id from test data")
test_id = raw_test['Id']

print("Deleting unnecessary columns out of test")
raw_test.drop('Dates', axis=1, inplace=True)
raw_test.drop('Id', axis=1, inplace=True)

print("Casting data")
columns = ['Year', 'Month', 'Hour', 'DayOfWeek', 'X', 'Y']
np_train = np.array(raw_train[columns]).astype(float)
np_train_category = np.array(raw_train_category).astype(float)
np_test = np.array(raw_test[columns]).astype(float)

print("Training")
clf = svm.SVC()
clf.fit(np_train, np_train_category)

print("Predicting")
test_classification = clf.predict(np_test)

print("Saving to csv")
np.savetxt('../results/sf-crime-submission.csv', test_classification, fmt='%f', delimiter=',')

print("Done!")

