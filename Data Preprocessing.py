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
traffic=traffic.drop_duplicates()



#%%

plt.plot(traffic.traffic_volume)
plt.show()


#%%

#Isolate after the break



d = datetime.datetime(2015, 6, 24)

print(d)

traffic_small = traffic[traffic.date_time>d].copy()

plt.figure(figsize=[11,3])
plt.plot(traffic_small.traffic_volume)
plt.show()



#%%


#OHE on the weather_main

for weather in traffic_small.weather_main.unique():
    traffic_small[weather+'-OHE'] = traffic_small.weather_main 
    traffic_small[weather+'-OHE'] = (traffic_small[weather+'-OHE']==weather)*1


#%%
#Make Holiday a boolean, rather than explicitly listing the holiday

traffic_small['holiday'] = traffic_small.holiday != 'None'

#%%

#Make a boolean if weekday
traffic_small['Weekday'] = traffic_small.date_time.dt.weekday
traffic_small['Weekday'] = traffic_small['Weekday'] < 5

#%%

#Now, collapse by date, since if a given hour had multiple weather conditions it was added as two seperate times

keepFirst=['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'traffic_volume', 'Weekday', 'weather_main', 'weather_description']

keepSum=[ 'Clear-OHE', 'Clouds-OHE', 'Rain-OHE', 'Haze-OHE', 'Thunderstorm-OHE', 'Mist-OHE',
       'Fog-OHE', 'Drizzle-OHE', 'Smoke-OHE', 'Snow-OHE', 'Squall-OHE']

#Crate the agg dictionaries
keepFirst={i:'first' for i in keepFirst}
keepSum={i:'sum' for i in keepSum}

#Merge the two dictionaries
keepFirst.update(keepSum)

#Group to gather the dates and collapse the OHE into a single row
traffic_small=traffic_small.groupby('date_time').agg(keepFirst).reset_index()

#%%

#Upsample the data to fill in the missing hours
#Pad to forward-fill the missing values (in a  2, NA, 5, it will fill as 2, 2, 5)
traffic_small=traffic_small.set_index('date_time').resample('H').pad()


#%%

#Group some of the Weather columns
    
traffic_small['Visibiliity-OHE'] = traffic_small['Fog-OHE'] + traffic_small['Mist-OHE'] + traffic_small['Haze-OHE'] + traffic_small['Smoke-OHE']

traffic_small['Precipitation-OHE'] = traffic_small['Thunderstorm-OHE'] + traffic_small['Mist-OHE'] + traffic_small['Drizzle-OHE'] + traffic_small['Snow-OHE'] + traffic_small['Rain-OHE'] + traffic_small['Squall-OHE']

traffic_small['Dry-OHE'] = traffic_small['Clear-OHE'] + traffic_small['Clouds-OHE']



#%%
traffic_small.to_csv('Reduced_Data.csv')




    




















