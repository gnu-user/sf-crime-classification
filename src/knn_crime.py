'''
Working from knn example posted at
https://www.kaggle.com/wawanco/sf-crime/k-nearest-neighbour/code
Original code only incorporated gps coodinates
Currently on the sf-crime competition he is ranked 631/754
'''
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


train = pd.read_csv('../data/train.csv', parse_dates=['Dates'])[['Dates', 'X', 'Y', 'Category']]


# Extract year, month, hour and remove dates
train['Year'], train['Month'], train['Hour'] = train.Dates.dt.year, train.Dates.dt.month, train.Dates.dt.hour
train.drop('Dates', axis=1, inplace=True)

train.X, train.Y = train.X.round(decimals=3), train.X.round(decimals=3)


print(train[:5])

y = train['Category'].astype('category')

# Submit for K=40

test = pd.read_csv('../data/test.csv', parse_dates=['Dates'])

# Extract year, month, hour and remove dates
test['Year'], test['Month'], test['Hour'] = test.Dates.dt.year, test.Dates.dt.month, test.Dates.dt.hour
test.drop('Dates', axis=1, inplace=True)
test.X, test.Y = test.X.round(decimals=3), test.X.round(decimals=3)

x_test = test[['Year', 'Month', 'Hour', 'X', 'Y']]

print("Training")
knn = KNeighborsClassifier(n_neighbors=40)
knn.fit(train[['Year', 'Month', 'Hour', 'X', 'Y']], y)

print("Predicting")
outcomes = knn.predict(x_test)

submit = pd.DataFrame({'Id': test.Id.tolist()})
for category in y.cat.categories:
    submit[category] = np.where(outcomes == category, 1, 0)

submit.to_csv('../results/k_nearest_neigbour.csv', index = False)