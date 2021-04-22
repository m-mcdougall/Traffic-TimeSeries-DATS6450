# -*- coding: utf-8 -*-
import pandas as pd
import os


#Set working directory
wd=os.path.abspath('C://Users//Mariko//Documents//GitHub//Traffic-TimeSeries-DATS6450//Data//')
os.chdir(wd)

pd.set_option('display.max_columns', None)

#Run Functions file that contains previously written functions.
runfile(os.path.abspath('..\Functions.py'))

#%%

## Start with loading the feature-selected Data
traffic_full = pd.read_csv('Modeling_Data.csv',index_col=0, parse_dates=[0])


#Isolate the target from the features
X=traffic_full.copy().drop('traffic_volume', axis=1)
Y=traffic_full.traffic_volume

#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle = False)

#%%

#========
# Basic Models
#--------------
# First, we're going to run the Basic models.
# This will give us a baseline for comparison
# when we make more intricate models later.
#========



method_dict={average_test:'Average', naive_test:'Naive', drift_test:'Drift', ses_test:'SES'}

for key in method_dict:
    
    avg = key(y_train, y_test)
    average_pred = avg[0]
    error =avg[1]
    error_2 = avg[2].values
    
    plot_prediction_method_axis(y_train, y_test, average_pred, error_2, method_str=method_dict[key])



#%%

#========
#Holt- Winter
#========

#Also a basic model, but requires extra steps.
#Requires strictly positive - there are two time-bins that have 0
#Add 1 to all values - should have minimal impact

#Start a holt-winter model
holtt=ets.ExponentialSmoothing(y_train.values+1,trend=None,damped_trend=False, seasonal='add', 
                               seasonal_periods=168).fit()
#Get predicted values
pred_y = holtt.forecast(steps=len(y_test.values))
pred_y=pd.DataFrame(pred_y).set_index(y_test.index)


#Calculate error
error = y_test+1- pred_y[0]
error_2 = error**2
    

plot_prediction_method_axis(y_train+1, y_test+1, pred_y, error_2, method_str='Holt-Winter')


#%%



#========
# Multiple Linear Regression Model
#--------------
# Start the more complicated models.
# Should perform better than the baseline 
# models established above.
#========


model = OLS(y_train,X_train).fit()
pred_y = model.predict(X_test)
forecast_errors = y_test-pred_y
error_2 = forecast_errors**2

plot_prediction_method_axis(y_train, y_test, pred_y, error_2, method_str='OLS Multiple Linear Regression')

print(model.summary())



#Now, for the Residuals

model = OLS(y_train,X_train).fit()
pred_y = model.predict(X_train)
residuals = y_train-pred_y
error_2 = residuals**2


plot_corr_full=run_auto_corr(residuals.values, lags=20, symmetrical=True)
plot_autocorrelation_simple(plot_corr_full, title_str='Autocorrelation of Residuals', original_array=residuals)


forecast_error_varience_calc(residuals, X_train)



print(model.summary())
print(f'\n======================\n')
print(f'The estimated mean of the forecast errors is:   {np.mean(forecast_errors):0.3f}')
print(f'The estimated variance of the forecast errors is:   {forecast_error_varience_calc(forecast_errors, X_test):0.3f}')
print()
print(f'The estimated mean of the residuals is:   {np.mean(residuals):0.3f}')
print(f'The estimated variance of the residuals is:   {forecast_error_varience_calc(residuals, X_train):0.3f}')

print(f'\n======================\n')
#Since the varience for the forecast error and residuals is similar, it shows the model adapts well to new information.

