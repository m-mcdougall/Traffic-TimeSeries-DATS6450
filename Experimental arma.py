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

y_diff=diff_seasonal_calc(y_train, interval=24)

cut_off=int(y_diff.shape[0]*0.8)


check=y_diff[:cut_off]
check_test=y_diff[cut_off::]

plt.plot(check)
plt.plot(check_test)
plt.show()

#%%
na=2
nb=0

model=statsmodels.tsa.arima.model.ARIMA(check, order=(na,0,nb),).fit()
na_params=model.params[0:na]*-1
nb_params=model.params[na::]

#1-step prediction
model_pred = model.predict(start=1, end=check.shape[0])
residuals= model.resid


#Plots the Training set
plt.figure(figsize=(8,6))
plt.plot(check, label='True Values')
plt.plot(model_pred, label='Predicted Values', alpha=0.9)
plt.title('Statsmodels ARMA Predicted Parameters Model')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.show()


#%%

plot_corr_full=run_auto_corr(residuals.values, lags=24, symmetrical=True)
plot_autocorrelation_simple(plot_corr_full, title_str='Autocorrelation of Residuals', original_array=residuals)

#%%


model_forecast = model.predict(start=check_test.index[0], end=check_test.index[-1])
#Plots the Testing set
plt.figure(figsize=(8,6))
plt.plot(check_test, label='True Values')
plt.plot(model_forecast[1::], label='Forecast Values', alpha=0.9)
plt.title('Statsmodels ARMA Predicted Parameters Model')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.show()

print('================')
Q=calc_Q_Score(residuals.values, y_train.values, lags=24, print_out=True)
deg_f=24-na-nb
print(f'Degrees of Freedom: {deg_f}')

if Q<chi2.ppf(0.95, deg_f):
    print('The Residuals are white')
else:
    print('The Residuals are not white')
print('================')




