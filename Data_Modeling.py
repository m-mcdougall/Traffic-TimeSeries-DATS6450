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


plot_corr_full=run_auto_corr(residuals.values, lags=24, symmetrical=True)
plot_autocorrelation_simple(plot_corr_full, title_str='Autocorrelation of Residuals', original_array=residuals)


forecast_error_varience_calc(residuals, X_train)

cal_q=calc_Q_Score(residuals.values, y_train.values, lags=24, print_out=False)


print(f'\n======================\n')
print(f'The estimated mean of the forecast errors is:   {np.mean(forecast_errors):0.3f}')
print(f'The estimated variance of the forecast errors is:   {forecast_error_varience_calc(forecast_errors, X_test):0.3f}')
print()
print(f'The estimated mean of the residuals is:   {np.mean(residuals):0.3f}')
print(f'The estimated variance of the residuals is:   {forecast_error_varience_calc(residuals, X_train):0.3f}')
print()
deg_f=24-3
print(f'The Q Score is: {cal_q:0.3f}')
print(f'The Q Crit  is: {chi2.ppf(0.95, deg_f):0.3f}\n')

if cal_q<chi2.ppf(0.95, deg_f):
    print('The Residuals are white')
else:
    print('The Residuals are not white')

print(f'\n======================\n')

#Since the varience for the forecast error and residuals is similar, it shows the model adapts well to new information.
#The ACF shows a lot of auto-correlation, meaning there is more information that the model is not capturing

#%%


#========
# ARMA and ARIMA and SARIMA model
#--------------
# Start work determining order of the ARIMA model
# First do a ACF/PACF and GPAC 
# From there, develop full models
#========


statstoolsACF_PACF(y_train, lags=24, title_str='')

createGPAC(y_train.values, equation_string='', j_val=12, k_val=12)


#Looking at the results of the gpac, either ARMA(2,0) or ARMA(6,0) could work
#Based on the pattern, seasonality likely a factor - SARIMA might be best

#%%

#ARMA(2,0)
na=2
nb=0

model=statsmodels.tsa.arima.model.ARIMA(y_train, order=(na,0,nb), freq='H').fit()
na_params=model.params[0:na]*-1
nb_params=model.params[na::]

#1-step prediction
model_pred = model.predict(start=1, end=y_train.shape[0])
residuals= model.resid


#Plots the Training set
plt.figure(figsize=(8,6))
plt.plot(y_train, label='True Values')
plt.plot(model_pred, label='Predicted Values', alpha=0.9)
plt.title('ARMA Predicted Parameters Model\n Training Set')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.show()



plot_corr_full=run_auto_corr(residuals.values, lags=24, symmetrical=True)
plot_autocorrelation_simple(plot_corr_full, title_str='Autocorrelation of Residuals', original_array=residuals)




#%%

#Generate the Confidence intervals for the Parameters

con_intervals=model.conf_int().drop(['const','sigma2'])

if na!=0:
    na_vals=model.arparams*-1
else:
    na_vals=0
na_con=con_intervals[0:na].values*-1


if nb!=0:
    nb_vals=model.maparams
else:
    nb_vals=0
nb_con=con_intervals[na::]

print(f'\n============\nConfidence Interval Results:\n============\n')
print('Na coeffs\n----------')
if len(na_con)==0:
    print('None')
else:
    [print(f'{na_vals[i]:0.3f}: {na_con[i][0]:0.3f} to {na_con[i][1]:0.3f}') for i in range(len(na_con))]

print('\nNb coeffs\n----------')
if len(nb_con)==0:
    print('None')
else:
    [print(f'{nb_vals[i]:0.3f}: {nb_con[i][0]:0.3f} to {nb_con[i][1]:0.3f}') for i in range(len(nb_con))]


#%%
    
#Generate the standard deviation

summary=model.summary().as_text()

#Extract the number of items used to make the STE
observations=int(summary[summary.find('No. Observations:')+len('No. Observations:'):summary.find('Model:')].strip())

#Extract the STE for each variable
summary_params=summary[summary.find('const'):summary.find('sigma2')].split('\n')
summary_params=summary_params[1:-1] #Remove the constant and sigma2 rows

#Extract the STE
collector=[]
for param in range(len(summary_params)):
    collector.append([i for i in summary_params[param].split(' ') if i != ''])

collector=pd.DataFrame(collector)
collector=collector.filter([0,1,2])


#Convert STE to STD
collector[2]=collector[2].astype(float)*np.sqrt(observations)


#Seperate and print the Standard Deviations

na_std=collector.iloc[0:na, :]
nb_std=collector.iloc[na::, :]

print(f'\n============\nStandard Deviation Results:\n============\n')
print('Na coeffs\n----------')
if na_std.shape[0]==0:
    print('None')
else:
    [print(f'{float(na_std.iloc[i,1])*-1:0.3f} STD: {na_std.iloc[i,2]:0.3f}') for i in range(na_std.shape[0])]

print('\nNb coeffs\n----------')
if nb_std.shape[0]==0:
    print('None')
else:
    [print(f'{float(nb_std.iloc[i,1])*-1:0.3f} STD: {nb_std.iloc[i,2]:0.3f}') for i in range(nb_std.shape[0])]

#%%


#Chi-squared Test

print('================')
Q=calc_Q_Score(residuals.values,  y_train.values, lags=24, print_out=False)
deg_f=24-na-nb

print(f'The Q Score is: {Q:0.3f}')
print(f'The Q Crit  is: {chi2.ppf(0.95, deg_f):0.3f}\n')

if Q<chi2.ppf(0.95, deg_f):
    print('The Residuals are white')
else:
    print('The Residuals are not white')
    
print(f'\nDegrees of Freedom: {deg_f}')
print('================')




#%%


#8 - Display poles and zeros (roots of numerator and roots of denominator)


print('\n Zero-Pole Cancellation\n')
params = model.params[1:-1]
poly_y=params[0:na].values
poly_e=params[na::].values

try:
    zeros = np.poly(poly_e)[1::]
    zero_pole_print(zeros)
except:
    print('(1-0)')
 
print('----------------')    
try:
    poles = np.poly(poly_y)[1::]
    zero_pole_print(poles)
except:
    print('(1-0)')    

    
  #%%  

































#%%
#Plots the Testing set
model_forecast = model.predict(start=y_train.shape[0], end=y_train.shape[0]+y_test.shape[0])

plt.figure(figsize=(8,6))
plt.plot(y_test, label='True Values')
plt.plot(model_forecast[1::], label='Forecast Values', alpha=0.9)
plt.title('ARMA Predicted Parameters Model\n Testing Set')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.show()