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

#
