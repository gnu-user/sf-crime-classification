import numpy as np
import pandas as pd
from sklearn import svm, grid_search
from sklearn.cross_validation import KFold
import csv
from sklearn.ensemble import RandomForestClassifier

from datetime import datetime, date



print("Reading in training data...")
# raw_train = np.array(list(csv.reader(open("../../CrimeData/train.csv", "rb"), delimiter=',')))
raw_train = pd.read_csv('../../CrimeData/train.csv')
print("done")

print("Reading in test data...")
# raw_test = np.array(list(csv.reader(open("../../CrimeData/test.csv", "rb"), delimiter=',')))
raw_test = pd.read_csv('../../CrimeData/test.csv')
print("done")

# ['Id' 'Dates' 'DayOfWeek' 'PdDistrict' 'Address' 'X' 'Y']
#print(raw_test[0])

# Header of file: Dates,Category,Descript,DayOfWeek,PdDistrict,Resolution,Address,X,Y
# Example contents: 015-05-13 23:53:00,WARRANTS,WARRANT ARREST,Wednesday,NORTHERN,"ARREST, BOOKED",
# OAK ST / LAGUNA ST,-122.425891675136,37.7745985956747

# extract test id column
test_id = raw_test['Id']

# create array of unique categories of crime
print("Finding unique categories")
categories = list(set(raw_train.Category))
print(categories)

# create array of unique dates in order
print("Finding unique dates")
train_dates = np.array([datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in raw_train.Dates])
test_dates = np.array([datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in raw_test.Dates])
dates = list(set(train_dates))
print(dates)

# create array of unique days of week in order
print("Finding unique days of the week")
days_of_week = list(set(raw_train['DayOfWeek']))

# X and Y are already float types, so leave them be

# create data frame representing the training data with data being converted to respective
# index in the corresponding array
print("Creating training data frame")
train_df_data = {'Dates': (dates.index(row) for row in train_dates),
              'DaysOfWeek': (days_of_week.index(row) for row in raw_train['DayOfWeek']),
              'X': raw_train['X'],
              'Y': raw_train['Y']}
train_df = pd.DataFrame(data=train_df_data, index=None)

# create data frame representing the test data with data being converted to respective
# index in the corresponding array
print("Creating test data frame")
test_df_data = {'Dates': (dates.index(row) for row in test_dates),
              'DaysOfWeek': (days_of_week.index(row) for row in raw_test['DayOfWeek']),
              'X': raw_test['X'],
              'Y': raw_test['Y']}
test_df = pd.DataFrame(data=test_df_data, index=None)


scores = []
k_fold_train = []
k_fold = KFold(n=len(raw_train), n_folds=3)

svc = svm.SVC()
gammas = {x: categories[x] for x in len(categories)}
clf = grid_search.GridSearchCV(estimator=svc, param_grid=dict(gamma=gammas), n_jobs=1)

train_class = (categories.index(row) for row in raw_train['Category'])

for k, (train, test) in enumerate(k_fold):
    print("Fold on k= {0} working.".format(k))
    clf.fit(train_df[train], train_class[train])
    scores.append(clf.score(train_df[test], train_class[test]))
    # k_fold_train.append(train)



print("K-Folds best score: {0}".format(np.amax(scores)))
