# -*- coding: utf-8 -*-


#To be run first.

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime




#Set working directory
wd=os.path.abspath('C://Users//Mariko//Documents//GitHub//Traffic-TimeSeries-DATS6450//Data//')
os.chdir(wd)

pd.set_option('display.max_columns', None)

#%%


## Start with loading the raw data
traffic = pd.read_csv('Metro_Interstate_Traffic_Volume.csv', parse_dates=[7])

#Set the datetime to the index
traffic.index=traffic.date_time


#Add day of week as factor
#Add binary weekday/weekend as factor
#%%

plt.plot(traffic.traffic_volume)
plt.show()


#%%

#Isolate after the break



d = datetime.datetime(2014, 12, 31)
print(d)

traffic_small = traffic[traffic.index>d]

plt.plot(traffic_small.traffic_volume)
plt.show()