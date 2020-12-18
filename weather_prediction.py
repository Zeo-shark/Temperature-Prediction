from __future__ import division
import csv, re, timeit, sys, math
from sklearn import datasets, linear_model, preprocessing, neural_network
from sklearn.utils import column_or_1d
from datetime import datetime, timedelta,date
from dateutil.relativedelta import relativedelta
import os
import errno
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from matplotlib import dates as mPlotDATEs

# num2 date() and date2num()


# http://stackoverflow.com/questions/4090383/plotting-unix-timestamps-in-matplotlib
# http://stackoverflow.com/questions/32728212/how-to-plot-timestamps-in-python-using-matplotlib
# http://stackoverflow.com/questions/8409095/matplotlib-set-markers-for-individual-points-on-a-line

rt_start= timeit.default_timer()

# claen log.txt first

directory= "logs"
try:
    os.makedirs(directory)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

log_timestr= datetime.now().strftime("%Y-%m-%d_%H%M%S")
with open("logs/log_" + log_timestr+ ".txt","w") as logfile:
    logfile.close()

#function definition
def print_data_type(x):
    for f in x.columns:
        print("f = {}".format(f))
        print(x[f].dtype)

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def RepresentsFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def dateToMinute(s, option):
    """
        time delta from the beginning of that year, unit = minutes
        s format = '%Y-%m-%d %H:%M'
        return type: int
        """
    if (isinstance(s, str)):
        time_obj= datetime.strptime(s, "%Y-%m-%d %H:%M")
    elif (isinstance(s, datetime)):
        time_obj= datetime.strptime( s.strftime('%Y-%m-%d %H:%M'),"%Y-%m-%d %H:%M" )
    else:
        raise SystemError("input is not a valid string/datetime obj!")

    if(option == 'year'):
        time_diff= time_obj - datetime(time_obj.year, 0o1, 0o1, 0, 0)
    elif(option== 'month'):
        time_diff= time_obj - datetime(time_obj.year, time_obj.month, 0o1, 0, 0)
    elif(option== 'day'):
        time_diff= time_obj - datetime(time_obj.year, time_obj.month, time_obj.day, 0,0)
    elif(option == 'hour'):
        time_diff= time_obj - datetime(time_obj.year, time_obj.month, time_obj.day,time_obj.hour, 0)
    else:
        raise SystemError("Option is not a valid string !")

    return int(time_diff.total_seconds()/60)

# do interpolate
def interpolate_df(df, features):
    df_re= df

    print("len(df.index) = {}".format(len(df.index)))

    # check all the data are float data and change data type to float64
    for col in features:
        # df[col]= df[col].astype(float)
        temp= df[df[col].isnull()]
        #print (test.head)
        print("===")
        #print(test.head(n=1))
        print("{} type is {}".format(col, df[col].dtype))
        print("{} type contain {} np.NaN".format(col, len(temp.index)))
        print("===")

    df_nan= df[df.isnull().any(axis=1)]
    print("len(df_nan.index) = {}".format(len(df_nan.index)))
    # df_nan.to_csv("df_nan.csv")

    df_nan.head(n=1)

    print("len(df.index) = {}".format(len(df.index)))
    # it could be use time as index and set method = 'time'
    # df.to_csv("df_before_interpolate.csv")
    # df[features] = df[features].interpolate(method='time')
    # df.loc[:, features] = df[features].interpolate(method='time')
    # somehow, df(input) will get updated even use inplace=False
    df_re.loc[:, features] = df[features].interpolate(method='time', inplace=False)
    # df.to_csv("df_after_interpolate.csv")
    # print("df = ")
    # print(df)

    #grab original nan values
    df_nan_interpolate= df.loc[ df_nan.index.values ]
    print("len(df_nan_interpolate.index) = {}".format(len(df_nan_interpolate.index)))
    df_nan_interpolate.to_csv("df_nan_interpolate.csv")

    if(df_re.notnull().all(axis=1).all(axis=0)):
        print("CHECK: There is no null value in df_re.")

    return df_re

# generating training and test dataset
def data_gen(df, targets, features, data_tr_yr_start, data_tr_yr_end, data_test_yr_start, data_test_yr_end):
    # reset index
    # df= df.reset_index(drop= True)
    df= df.set_index("DATE")
    #prepare training date
    data_start= datetime(data_tr_yr_start ,1, 1, 1, 0, 0)
    data_end =   datetime(data_test_yr_end, 12, 31, 23, 59, 59)
    df_train=  df.loc[(df.index> data_start ) & (df.index <= data_end ), :]


    # do interpolate on training set only
    df_train= interpolate_df(df_train, features)
    df_train.to_csv('df_train_clean.csv')

    X_train= df_train[features]
    y_train= df_train[targets]

    #prepare the test data
    data_start= datetime(data_test_yr_start, 1, 1, 0, 0, 0)
    data_end=   datetime(data_test_yr_end, 12, 31, 23, 59, 59)
    df_test= df.loc[(df.index> df.start)  & (df.index<= data_end), :]

    # drop NaN number rows of test set
    (row_old, col_old) = df_test.shape
    print("Before drop NaN number of test set, df_test.shape = {}".format(df_test.shape))
    df_test = df_test[df_test.notnull().all(axis=1)]
    (row, col) = df_test.shape
    print("After drop NaN number of test set, df_test.shape = {}".format(df_test.shape))
    print("Drop rate = {0:.2f} ".format(float(1 - (row / row_old))))

    df_test.to_csv('df_test_clean.csv')
    X_test = df_test[features]
    y_test = df_test[targets]

    # normalization and scale for training/test set
    # use robust_scaler to avoid misleading outliers
    # scaler = preprocessing.StandardScaler()
    # use robust_scaler to avoid misleading outliers
    scaler = preprocessing.RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (X_train, y_train, X_test, y_test)


def normalisation(df_train, df_test, targets, features):

    #do interpolate on training data only
    df_train_local=  interpolate_df(df_train, features)

    X_train= df_train_local[features]
    y_train= df_train_local[targets]

    X_test= df_test[features]
    y_test= df_test[targets]

    #normalization and scale for training/test set
    # use robust_scaler to avoid misleading outliers
    # scaler= preprocessing.StandardScaler()
    # use robust_scaler to avoid misleading outliers
    scaler= preprocessing.RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test =  scaler.transform(X_test)

    return (X_train, y_train, X_test, y_test)

#plot y_test
def plot_y_test(regr, X_test, y_test, ask_user):
    (r_test, c_test)= X_test.shape

    # for i in range(c_test):
    #     plt.scatter(X_test[:, i], y_test)
    #     plt.plot(X_test[:, i], regr.predict(X_test), color='blue', linewidth=3)

    y_predict = regr.predict(X_test)
    # print("==> y_test type = {}".format(type(y_test)) )
    # print("y_test.index = {}".format(y_test.index))
    # print("y_test = {}".format(y_test) )
    # print("y_predict = {}".format(y_predict) )
    df_plot = y_test
    # print(df_plot)
    # print("DATE")
    # print("#########################")

    #df_plot = df_plot.reset_index(level=['DATE'])
    df_plot = df_plot.reset_index(level= ['DATE'])
    #df_plot.iloc()
    df_plot.loc[:, 'predict_temp_C'] = y_predict
    # shift back to raw DATE time 1day_later
    df_plot.loc[:, "raw_DATE"] = df_plot['DATE'].apply(lambda time_obj: time_obj + relativedelta(days=1))
    df_plot.rename(columns={'1days_later_temp_C': 'raw_temp_C', 'DATE': 'label_DATE'}, inplace=True)

    df_plot = df_plot.set_index("raw_DATE")
    # print(df_plot)
    # print("#########################")


