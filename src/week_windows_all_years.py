import numpy as np 
import pandas as pd
from sklearn import svm
from datetime import datetime, date
from sklearn.preprocessing import StandardScaler


def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


print("Reading in training data")
# raw_train = np.array(list(csv.reader(open("../data/train.csv", "rb"), delimiter=',')))
raw_train = pd.read_csv('../data/train.csv', parse_dates=['Dates'])

print("Reading in test data")
# raw_test = np.array(list(csv.reader(open("../data/test.csv", "rb"), delimiter=',')))
raw_test = pd.read_csv('../data/test.csv', parse_dates=['Dates'])

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

raw_train['Year'], raw_train['Week'], raw_train['Hour'] = raw_train.Dates.dt.year, raw_train.Dates.dt.week, raw_train.Dates.dt.hour
raw_test['Year'], raw_test['Week'], raw_test['Hour'] = raw_test.Dates.dt.year, raw_test.Dates.dt.week, raw_test.Dates.dt.hour


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
    year_train_df = raw_train.where(raw_train.Year == year)[['Week', 'Hour', 'DayOfWeek', 'X', 'Y', 'Category']]
    year_test_df = raw_test.where(raw_test.Year == year)[['Week', 'Hour', 'DayOfWeek', 'X', 'Y']]

    # loop over months, and remove nan value
    distinct_weeks = year_train_df.Week.unique()
    distinct_weeks = distinct_weeks[~np.isnan(distinct_weeks)]
    for week in range(int(distinct_weeks.max()), int(distinct_weeks.min())-1, -2):
        # create subset of data of current year
        print("Creating subset of data for week {0}".format(week))
        week_train_df = year_train_df.where(year_train_df.Week == week)
        week_train_category = week_train_df['Category']
        week_train_df = week_train_df[['Hour', 'DayOfWeek', 'X', 'Y']]
        week_test_df = year_test_df.where(year_test_df.Week == week-1)[['Hour', 'DayOfWeek', 'X', 'Y']]
        

        
        print("Training on {0} rows".format(len(week_train_df)))
        #filter NaN vaule rows
        available_train = pd.notnull(week_train_category)
        available_test = pd.notnull(week_test_df['X'])
        week_train_df = week_train_df[available_train]
        week_train_category = week_train_category[available_train]
        week_test_df = week_test_df[available_test]

        #standlize the data
        week_train_df, scaler = preprocess_data(week_train_df)
        week_test_df, _ = preprocess_data(week_test_df, scaler)
        
        #train model
        clf = svm.SVC(probability=True)
        clf.fit(week_train_df, week_train_category)
        print("Predicting on {0} rows".format(week_test_df))
        temp_predictions = clf.predict_proba(week_test_df)
        
        #concatenates results
        temp_predictions = pd.DataFrame(temp_predictions)
        temp_predictions.columns = clf.classes_
        #print temp_predictions
        predictions = predictions.append(temp_predictions, ignore_index=True)

# address NaN values
predictions[pd.isnull(predictions)] = 0


print("Saving to csv")
predictions.to_csv('../results/sf-crime-submission.csv', index_label="Id", na_rep=0)


print("Done!")

