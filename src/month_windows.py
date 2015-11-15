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
days_of_week = raw_train.DayOfWeek.unique()
days_of_week_dict = {days_of_week[x]: float(x) for x in range(len(days_of_week))}
raw_train.DayOfWeek = raw_train.DayOfWeek.map(days_of_week_dict)
raw_test.DayOfWeek = raw_test.DayOfWeek.map(days_of_week_dict)

print("Mapping categories")

categories = raw_train.Category.unique()
categories_dict = {categories[x]: float(x) for x in range(len(categories))}
# raw_train_category = raw_train.Category.map(categories_dict)

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
#raw_train.drop('Category', axis=1, inplace=True)

# extract test id column
print("Extracting Id from test data")
test_id = raw_test['Id']

print("Deleting unnecessary columns out of test")
raw_test.drop('Dates', axis=1, inplace=True)
raw_test.drop('Id', axis=1, inplace=True)




# Both the training and test data goes from most recent to oldest
# want to train over month to predict month
# Current layout of train/test columns is:
# ['Year', 'Month', 'Hour', 'DayOfWeek', 'X', 'Y', 'Category']
# ***test does not include category

# create predictions data frame to append to
predictions = pd.DataFrame()


print("Starting the training/testing loop")
distinct_years = raw_train.Year.unique()
for year in range(distinct_years.max(),distinct_years.min() - 1, -1):

    # create subset of data of current year
    print("Creating subset of data for year {0}".format(year))
    year_train_df = raw_train.where(raw_train.Year == year)[['Month', 'Hour', 'DayOfWeek', 'X', 'Y', 'Category']]
    year_test_df = raw_test.where(raw_test.Year == year)[['Month', 'Hour', 'DayOfWeek', 'X', 'Y']]

    # loop over months, and remove nan value
    distinct_months = year_train_df.Month.unique()
    distinct_months = distinct_months[~np.isnan(distinct_months)]
    for month in range(int(distinct_months.max()), int(distinct_months.min()) - 1 , -1):
        # create subset of data of current year
        print("Creating subset of data for month {0}".format(month))
        month_train_df = year_train_df.where(year_train_df.Month == month)
        month_train_category = month_train_df['Category']
        month_train_df = month_train_df[['Hour', 'DayOfWeek', 'X', 'Y']]
        month_test_df = year_test_df.where(year_test_df.Month == month)[['Hour', 'DayOfWeek', 'X', 'Y']]

        print("Training on {0} rows".format(len(month_train_df)))
        #filter NaN vaule rows
        available_train = pd.notnull(month_train_category)
        available_test = pd.notnull(month_test_df['X'])
        month_train_df = month_train_df[available_train]
        month_train_category = month_train_category[available_train]
        month_test_df = month_test_df[available_test]
        
        #train model
        clf = svm.SVC()
        clf.fit(month_train_df, month_train_category)
        print("Predicting on {0} rows".format(month_test_df))
        predictions = [predictions, pd.DataFrame(clf.predict(month_test_df))]
        predictions = pd.concat(predictions)

print("Saving to csv")
np.savetxt('../results/sf-crime-submission.csv', predictions, fmt='%f', delimiter=',')

'''
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
'''
print("Done!")

