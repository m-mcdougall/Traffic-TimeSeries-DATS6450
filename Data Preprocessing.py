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



d = datetime.datetime(2015, 6, 24)
print(d)

traffic_small = traffic[traffic.index>d].copy()

plt.plot(traffic_small.traffic_volume)
plt.show()

#%%


#OHE on the weather_main

for weather in traffic_small.weather_main.unique():
    traffic_small[weather+'-OHE'] = traffic_small.weather_main 
    traffic_small[weather+'-OHE'] = traffic_small[weather+'-OHE']==weather


#%%
    
#Collapse some of the OHE columns
    
traffic_small['Visibiliity-OHE'] = traffic_small['Fog-OHE'] + traffic_small['Mist-OHE'] + traffic_small['Haze-OHE'] + traffic_small['Smoke-OHE']

traffic_small['Precipitation-OHE'] = traffic_small['Thunderstorm-OHE'] + traffic_small['Mist-OHE'] + traffic_small['Drizzle-OHE'] + traffic_small['Snow-OHE'] + traffic_small['Rain-OHE'] + traffic_small['Squall-OHE']

traffic_small['Dry-OHE'] = traffic_small['Clear-OHE'] + traffic_small['Clouds-OHE']
#%%
#Make Holiday a boolean, rather than explicitly listing the holiday

traffic_small['holiday'] = traffic_small.holiday != 'None'

#%%

#Make a boolean if weekday
traffic_small['Weekday'] = traffic_small.index.weekday
traffic_small['Weekday'] = traffic_small['Weekday'] < 5

#%%
traffic_small.to_csv('Reduced_Data.csv')

