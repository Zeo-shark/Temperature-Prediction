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


