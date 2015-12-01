import pandas as pd
import numpy as np
from datetime import datetime
# from sklearn.cross_validation import train_test_split
# from sklearn.grid_search import GridSearchCV
# import matplotlib.pylab as plt
# from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
# from sklearn.metrics import make_scorer
from sklearn.cross_validation import StratifiedShuffleSplit
# from matplotlib.colors import LogNorm
# from sklearn.decomposition import PCA
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import Callback
from copy import deepcopy
from collections import deque
from time import time
from datetime import timedelta
import warnings

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--nodes', type=int, default=128)
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])

args = parser.parse_args()

# NEURAL NET PARAMETERS
N_EPOCHS = args.epochs
N_HN = args.nodes
N_LAYERS = args.layers
DP = args.dropout
OPTIMIZER = args.optimizer


class BetterLogger(Callback):
    def __init__(self, monitor='loss', patience=100, verbose=1):
        super(Callback, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.best = np.Inf
        self.wait = 0

    def on_train_begin(self, logs={}):
        self.nb_epoch = self.params['nb_epoch']
        self.times = deque(maxlen=10)
        self.epoch = []
        self.history = {}

    def on_epoch_begin(self, epoch, logs={}):
        self.seen = 0
        self.totals = {}
        self.start_time = time()

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)
        self.seen += batch_size
        for k, v in logs.items():
            if k in self.totals:
                self.totals[k] += v * batch_size
            else:
                self.totals[k] = v * batch_size

    def on_epoch_end(self, epoch, logs={}):
        self.epoch.append(epoch)
        for k, v in self.totals.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v / self.seen)

        for k, v in logs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)

        dt = time() - self.start_time
        self.times.append(dt)
        if self.verbose:
            print('Epoch: {:5d}/{}\tTime: {:.1f}s\tLoss: {:.4f}\tETA: {}'
                  .format(epoch + 1, self.nb_epoch, dt,
                          self.history['loss'][-1],
                          str(timedelta
                              (seconds=int((self.nb_epoch - epoch) *
                                           (sum(self.times) /
                                            len(self.times)))))))

        current = self.history[self.monitor][-1]
        if current is None:
            warnings.warn("Early stopping requires %s available!"
                          .format(self.monitor), RuntimeWarning)

        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping" % (epoch))
                self.model.stop_training = True
            self.wait += 1


def parse_time(x):
    DD = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    time = DD.hour   # *60+DD.minute
    day = DD.day
    month = DD.month
    year = DD.year
    return time, day, month, year


def get_season(x):
    summer = 0
    fall = 0
    winter = 0
    spring = 0
    if (x in [5, 6, 7]):
        summer = 1
    if (x in [8, 9, 10]):
        fall = 1
    if (x in [11, 0, 1]):
        winter = 1
    if (x in [2, 3, 4]):
        spring = 1
    return summer, fall, winter, spring


def build_and_fit_model(X_train, y_train, X_test=None, y_test=None, hn=32,
                        dp=0.5, layers=1, epochs=1, batches=64, verbose=0,
                        optimizer='adam'):
    input_dim = X_train.shape[1]
    output_dim = len(labels_train.unique())
    Y_train = np_utils.to_categorical(y_train
                                      .cat
                                      .rename_categories(
                                          range(len(y_train.unique()))))
    # print output_dim
    model = Sequential()
    model.add(Dense(hn, input_dim=input_dim, init='glorot_uniform'))
    model.add(PReLU(input_shape=(hn, )))
    model.add(Dropout(dp))

    for i in range(layers):
        model.add(Dense(hn, init='glorot_uniform'))
        model.add(PReLU(input_shape=(hn, )))
        model.add(BatchNormalization(input_shape=(hn, )))
        model.add(Dropout(dp))

    model.add(Dense(output_dim, init='glorot_uniform'))
    model.add(Activation('softmax'))
    if optimizer == 'sgd':
        optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    if X_test is not None:
        Y_test = np_utils.to_categorical(y_test
                                         .cat
                                         .rename_categories(
                                             range(len(y_test.unique()))))
        fitting = model.fit(X_train, Y_train, nb_epoch=epochs,
                            batch_size=batches, verbose=verbose,
                            validation_data=(X_test, Y_test))
        test_score = log_loss(y_test, model.predict_proba(X_test, verbose=0))
    else:
        model.fit(X_train, Y_train, nb_epoch=epochs, batch_size=batches,
                  verbose=0, callbacks=[BetterLogger()])
        fitting = 0
        test_score = 0
    return test_score, fitting, model


def parse_data(df, logodds, logoddsPA):
    feature_list = df.columns.tolist()
    if "Descript" in feature_list:
        feature_list.remove("Descript")
    if "Resolution" in feature_list:
        feature_list.remove("Resolution")
    if "Category" in feature_list:
        feature_list.remove("Category")
    if "Id" in feature_list:
        feature_list.remove("Id")
    cleanData = df[feature_list]
    cleanData.index = range(len(df))
    print("Creating address features")
    address_features = cleanData["Address"].apply(lambda x: logodds[x])
    address_features.columns = ["logodds"+str(x) for x in
                                range(len(address_features.columns))]
    print("Parsing dates")
    (cleanData["Time"], cleanData["Day"],
     cleanData["Month"], cleanData["Year"]) = zip(*cleanData["Dates"]
                                                  .apply(parse_time))
    # dummy_ranks_DAY = pd.get_dummies(cleanData['DayOfWeek'], prefix = 'DAY')
    # days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
    #         'Friday', 'Saturday', 'Sunday']
    # cleanData["DayOfWeek"] = cleanData["DayOfWeek"].apply(lambda x:
    #                                                       days.index(x)/float(len(days)))
    print("Creating one-hot variables")
    dummy_ranks_PD = pd.get_dummies(cleanData['PdDistrict'], prefix='PD')
    dummy_ranks_DAY = pd.get_dummies(cleanData["DayOfWeek"], prefix='DAY')
    cleanData["IsInterection"] = (cleanData["Address"]
                                  .apply(lambda x: 1 if "/" in x else 0))
    cleanData["logoddsPA"] = cleanData["Address"].apply(lambda x: logoddsPA[x])
    print("droping processed columns")
    cleanData = cleanData.drop("PdDistrict", axis=1)
    cleanData = cleanData.drop("DayOfWeek", axis=1)
    cleanData = cleanData.drop("Address", axis=1)
    cleanData = cleanData.drop("Dates", axis=1)
    feature_list = cleanData.columns.tolist()
    print("joining one-hot features")
    features = (cleanData[feature_list]
                .join(dummy_ranks_PD.ix[:, :])
                .join(dummy_ranks_DAY.ix[:, :])
                .join(address_features.ix[:, :]))
    print("creating new features")
    features["IsDup"] = (pd.Series(features.duplicated() |
                                   features.duplicated(keep='last'))
                         .apply(int))
    features["Awake"] = (features["Time"]
                         .apply(lambda x: 1
                                if (x == 0 or (x >= 8 and x <= 23))
                                else 0))
    (features["Summer"], features["Fall"],
     features["Winter"], features["Spring"]) = (zip(*features["Month"]
                                                    .apply(get_season)))
    if "Category" in df.columns:
        labels = df["Category"].astype('category')
        # label_names=labels.unique()
        # labels=labels.cat.rename_categories(range(len(label_names)))
    else:
        labels = None
    return features, labels

# Import training data
print("PROCESSING TRAINING DATA...")
trainDF = pd.read_csv("../data/train.csv")

# Clean up wrong X and Y values (very few of them)
xy_scaler = StandardScaler()
xy_scaler.fit(trainDF[["X", "Y"]])
trainDF[["X", "Y"]] = xy_scaler.transform(trainDF[["X", "Y"]])
trainDF = trainDF[abs(trainDF["Y"]) < 100]
trainDF.index = range(len(trainDF))

# This takes a while...
addresses = sorted(trainDF["Address"].unique())
categories = sorted(trainDF["Category"].unique())
C_counts = trainDF.groupby(["Category"]).size()
A_C_counts = trainDF.groupby(["Address", "Category"]).size()
A_counts = trainDF.groupby(["Address"]).size()
logodds = {}
logoddsPA = {}
MIN_CAT_COUNTS = 2
default_logodds = (np.log(C_counts/len(trainDF)) -
                   np.log(1.0-C_counts/float(len(trainDF))))
for addr in addresses:
    PA = A_counts[addr]/float(len(trainDF))
    logoddsPA[addr] = np.log(PA)-np.log(1.-PA)
    logodds[addr] = deepcopy(default_logodds)
    for cat in A_C_counts[addr].keys():
        if (((A_C_counts[addr][cat] > MIN_CAT_COUNTS) and
             (A_C_counts[addr][cat] < A_counts[addr]))):
            PA = A_C_counts[addr][cat]/float(A_counts[addr])
            logodds[addr][categories.index(cat)] = np.log(PA)-np.log(1.0-PA)
    logodds[addr] = pd.Series(logodds[addr])
    logodds[addr].index = range(len(categories))

features, labels = parse_data(trainDF, logodds, logoddsPA)

# num_feature_list = ["Time", "Day", "Month", "Year", "DayOfWeek"]
collist = features.columns.tolist()
scaler = StandardScaler()
scaler.fit(features)
features[collist] = scaler.transform(features)

sss = StratifiedShuffleSplit(labels, train_size=0.5)
for train_index, test_index in sss:
    features_train, features_test = (features.iloc[train_index],
                                     features.iloc[test_index])
    labels_train, labels_test = labels[train_index], labels[test_index]
features_test.index = range(len(features_test))
features_train.index = range(len(features_train))
labels_train.index = range(len(labels_train))
labels_test.index = range(len(labels_test))
features.index = range(len(features))
labels.index = range(len(labels))


# # Run the data through the NN
# print("INITIAL MODEL TRAINING...")
# score, fitting, model = build_and_fit_model(features_train.as_matrix(),
#                                             labels_train,
#                                             X_test=features_test.as_matrix(),
#                                             y_test=labels_test,
#                                             hn=N_HN, layers=N_LAYERS,
#                                             epochs=N_EPOCHS, verbose=2, dp=DP)


# # Print the results
# print("all", log_loss(labels, model.predict_proba(features.as_matrix(),
#                                                   verbose=0)))
# print("train", log_loss(labels_train, model.predict_proba(features_train
#                                                           .as_matrix(),
#                                                           verbose=0)))
# print("test", log_loss(labels_test, model.predict_proba(features_test
#                                                         .as_matrix(),
#                                                         verbose=0)))

# Train the final model
print("FINAL MODEL TRAINING...")
score, fitting, model = build_and_fit_model(features.as_matrix(), labels,
                                            hn=N_HN,
                                            layers=N_LAYERS, epochs=N_EPOCHS,
                                            verbose=2, dp=DP,
                                            optimizer=OPTIMIZER)

# Results from final model
print("all", log_loss(labels, model.predict_proba(features.as_matrix(),
                                                  verbose=0)))
print("train", log_loss(labels_train, model.predict_proba(features_train
                                                          .as_matrix(),
                                                          verbose=0)))
print("test", log_loss(labels_test, model.predict_proba(features_test
                                                        .as_matrix(),
                                                        verbose=0)))


# Load in the test data
print("PROCESSING TEST DATA...")
testDF = pd.read_csv("../data/test.csv")
testDF[["X", "Y"]] = xy_scaler.transform(testDF[["X", "Y"]])

# set outliers to 0
testDF["X"] = testDF["X"].apply(lambda x: 0 if abs(x) > 5 else x)
testDF["Y"] = testDF["Y"].apply(lambda y: 0 if abs(y) > 5 else y)

# Process the test data, in particular the address
new_addresses = sorted(testDF["Address"].unique())
new_A_counts = testDF.groupby("Address").size()
only_new = set(new_addresses+addresses)-set(addresses)
only_old = set(new_addresses+addresses)-set(new_addresses)
in_both = set(new_addresses).intersection(addresses)
for addr in only_new:
    PA = new_A_counts[addr]/float(len(testDF)+len(trainDF))
    logoddsPA[addr] = np.log(PA)-np.log(1.-PA)
    logodds[addr] = deepcopy(default_logodds)
    logodds[addr].index = range(len(categories))
for addr in in_both:
    PA = (A_counts[addr]+new_A_counts[addr])/float(len(testDF)+len(trainDF))
    logoddsPA[addr] = np.log(PA)-np.log(1.-PA)

# Parse data, takes a while
features_sub, _ = parse_data(testDF, logodds, logoddsPA)
# scaler.fit(features_test)

# Get the final features and classify the test data using NN
print("CLASSIFYING TEST DATA USING NN...")
collist = features_sub.columns.tolist()
features_sub[collist] = scaler.transform(features_sub[collist])
predDF = pd.DataFrame(model.predict_proba(features_sub.as_matrix(), verbose=0),
                      columns=sorted(labels.unique()))
print("SAVING FINAL RESULTS...")
name = ('sf-crime-{}-layer-{}-node-{}-epoch-{}.csv'
        .format(N_LAYERS, N_HN, N_EPOCHS, OPTIMIZER))
predDF.to_csv("../results/{}".format(name),
              index_label="Id", na_rep="0")
