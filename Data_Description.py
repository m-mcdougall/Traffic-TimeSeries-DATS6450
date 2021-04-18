# -*- coding: utf-8 -*-
#To be run Second.

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from statsmodels.tsa.stattools import adfuller
import seaborn as sns


#Set working directory
wd=os.path.abspath('C://Users//Mariko//Documents//GitHub//Traffic-TimeSeries-DATS6450//Data//')
os.chdir(wd)

pd.set_option('display.max_columns', None)

#Run Functions file that contains previously written functions.
runfile(os.path.abspath('..\Functions.py'))


#%%

## Start with loading the processed data
traffic_full = pd.read_csv('Reduced_Data.csv',index_col=0, parse_dates=[0])

#Set the datetime to the index
traffic = traffic_full.traffic_volume
traffic_df=pd.DataFrame(traffic)
lags=24

#%%

#A - Plot the data
plt.figure(figsize=[14,5], )
plt.plot(traffic_df,label='Traffic Volume')
plt.title('Hourly Traffic Volume')
plt.xlabel('Time')
plt.ylabel('Number of Cars')
plt.legend()
plt.show()

#%%
## B - Plot the ACF and PACF
statstoolsACF_PACF(traffic, lags=lags, title_str='Traffic Volume')




#%%


#Do a correlation heatmap of all features
corr=traffic_full.corr()
fig, ax = plt.subplots(figsize=[11,9])
sns.heatmap(corr, vmax=1, vmin=-1, center=0, cmap='vlag', annot=True, fmt='0.1f',ax=ax)
plt.title("Correlation of Traffic Data")
plt.show()



#%%

#Confirm that there are no missing values
traffic_full.isnull().sum()


#%%

#Split into Training and Testing

#Isolate the target from the features
X=traffic_full.copy().drop('traffic_volume', axis=1)
Y=traffic_full.traffic_volume

#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle = False)


#%%

#Transform to Dataframe
y_train_df=pd.DataFrame(y_train)

#Show that the data is stationary    
x=sub_sequence_stats(y_train_df, 'traffic_volume')

#Plot the semi-rolling varience and means
plt.figure()
plt.subplot(211)
plt.plot(x.Means,  label='Means')
plt.ylabel('Mean Traffic Density')
plt.legend()
plt.subplot(212)
plt.plot(x.Varience, color='orange', label='Varience')
plt.ylabel('Varience in Traffic Density')
plt.xlabel('Datapoints in Subsequence')
plt.legend()
plt.suptitle(f'Rolling Means and Varience of Traffic Density', y=1.03)
plt.tight_layout()
plt.show()


#Do ADF Test
ADF_col(y_train.values, 'Traffic Volume')

#%%

"""
Plots the STL decomposition
"""

#One Week
model = STL(y_train_df.traffic_volume, period=24)
res=model.fit()
t=res.trend
s=res.seasonal
r=res.resid

#Temporary plotsize adjust
with plt.rc_context():
    plt.rc("figure", figsize=(10,6))
    fig=res.plot()
    plt.xlabel('Time (Hourly)')
    plt.suptitle(f'STL Decomposition of Traffic Data - Period 1 Day', fontsize=14, y=1.05)
    plt.show()



"""
Calculate the strength of Seasonality and Trend in the original data
"""

print('\n --------------')
#The strength of the trend
strength_trend = np.max([0, 1-(r.var()/(t+r).var())])
print(f'\n The strength of trend for this data set is {strength_trend:0.3f}')

#The strength of the seasonality
strength_seasonal = np.max([0, 1-(r.var()/(s+r).var())])
print(f' The strength of seasonality for this data set is {strength_seasonal:0.3f}')
print('\n --------------')





"""
Plots the Seasonal Adjustment of the original data
"""


adjusted_seasonal = y_train_df.traffic_volume - s #Adjust for seasonality
adjusted_trend = y_train_df.traffic_volume - t #Adjust for trend

with plt.rc_context():
    plt.rc("figure", figsize=(10,6))
    y_train_df.traffic_volume.plot(label='Original Data')
    adjusted_seasonal.plot(label='Seasonally Adjusted Data')
    plt.xlabel('Time (Hourly)')
    plt.ylabel('Traffic Volume')
    plt.title(f'Seasonally Adjusted and Original Sales Data\n Strength of Seasonality:{strength_seasonal:0.3f}', fontsize=14,)
    plt.legend(loc='upper left')
    plt.show()


with plt.rc_context():
    plt.rc("figure", figsize=(10,6))
    y_train_df.traffic_volume.plot(label='Original Data')
    adjusted_trend.plot(label='De-Trended Data', color='green')
    plt.xlabel('Time (Hourly)')
    plt.ylabel('Traffic Volume')
    plt.title(f'De-Trended and Original Sales Data\n Strength of Trend:{strength_trend:0.3f}', fontsize=14,)
    plt.legend(loc='upper left')
    plt.show()
    

#%%

#==================
#
#Feature Selection
#    
#==================


#To do feature selection, you need the full dataset.
    
df_feature_select = traffic_full.copy()

# First, drop the string columns - they have already been encorpertated as OHE columns
df_feature_select=df_feature_select.drop(['weather_main', 'weather_description'], axis=1)


#Perform an SVD analysis on the original feature space.
s,d,v = la.svd(df_feature_select.values.astype(float))

print(f"\nSingularValues:\n{d}\n")
#Since three of the values are extremely close to 0, at least one feature is highly correlated.



#Perform a condition number check
print(f"Condition Number\n{la.cond(df_feature_select.values.astype(float))}\n")
#Since the condition number is >100, there is co-linearity, and since it is >1000, the colinearity is severe.


#%%


#Begin removing features to acheive a higher Adj R-squared and reduce colinearity
#Isolate the target from the features
X=df_feature_select.copy().drop(['traffic_volume', ], axis=1).astype(float) #Adj. R-squared  0.760
Y=df_feature_select.traffic_volume

#Drop Features here
#-----------------------    
X=X.drop(['Rain-OHE'], axis=1) #Adj. R-squared  0.760
X=X.drop(['Precipitation-OHE'], axis=1) #Adj. R-squared  0.760
X=X.drop(['Squall-OHE'], axis=1) #Adj. R-squared  0.760
X=X.drop(['Fog-OHE'], axis=1) #Adj. R-squared  0.760
X=X.drop(['Smoke-OHE'], axis=1) #Adj. R-squared  0.760
X=X.drop(['rain_1h'], axis=1) #Adj. R-squared  0.760 
X=X.drop(['snow_1h'], axis=1) #Adj. R-squared  0.760
X=X.drop(['Snow-OHE'], axis=1) #Adj. R-squared  0.760
X=X.drop(['Mist-OHE'], axis=1) #Adj. R-squared  0.760
X=X.drop(['Clear-OHE'], axis=1) #Adj. R-squared  0.760
X=X.drop(['holiday'], axis=1) #Adj. R-squared  0.759
X=X.drop(['Haze-OHE'], axis=1) #Adj. R-squared  0.758
X=X.drop(['Drizzle-OHE'], axis=1) #Adj. R-squared  0.758
X=X.drop(['Thunderstorm-OHE'], axis=1) #Adj. R-squared  0.758
X=X.drop(['Dry-OHE'], axis=1) #Adj. R-squared  0.758
X=X.drop(['Visibiliity-OHE'], axis=1) #Adj. R-squared  0.757
X=X.drop(['Clouds-OHE'], axis=1) #Adj. R-squared  0.754



#Further variable removal lead to large drop in Adjusted R-Squared 
# Feature selection complete.
#-----------------------
#X=X.drop(['Weekday'], axis=1) #Adj. R-squared  0.744
#X=X.drop(['temp'], axis=1) #Adj. R-squared  0.659

#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=False )
model = OLS(y_train,X_train).fit()
pred = model.predict(X_test)
print(model.summary())


#%%

#Check the SVD and Condition numbers of the reduced feature space

#SVD and condition numbers
s,d,v = la.svd(X)

print(f"\nSingularValues:\n{d}\n")
print(f"Condition Number\n{la.cond(X)}\n")

#All Singular values are >0, therefore no features are highly correlated.
#Since the condition number is <100, there is no co-linearity between features.


#%%

#Save the reduced feature space for the modeling steps.

#Merge back ito one Dataframe
modeling_data=pd.concat([X, Y], axis=1)

#Save
modeling_data.to_csv('Modeling_Data.csv')





