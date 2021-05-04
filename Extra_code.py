# -*- coding: utf-8 -*-
#%%


"""

ARMA(2,0)

"""
#One step ahead model - Manual

"""
Do the 50 Step prediction, and plot
"""

#Input variables
steps=50
furthest_back=2

#Collector for final set
varience_pred = pd.Series()


#A series that contains all values needed for the predictions.
#This includes the past values and the predicted values
values=pd.Series(np.zeros((steps+furthest_back+1)))

#Re-index for easy access
values.index=[i for i in range(-1*furthest_back, steps+1)]

#Seed the past values
for i in range(-1-furthest_back,0):
    values[i+1] = y_train[i]

#Now, incrementally make predictions    
for i in range(1,steps+1):
    values[i] = 1.374462*values[i-1] -0.520772*values[i-2]

#Include only the predicted values (Exclude the training values)
pred_val=values[furthest_back+1:steps+furthest_back+1]

#Return the predicted values
varience_pred=varience_pred.append(pred_val)

print(f'The Test varience is {y_test.var():0.2f}')
print(f'The Predicted varience is {varience_pred.var():0.2f}')
print(f'The Varience ratio is {y_test.var()/varience_pred.var():0.2f}')


#Create a plot of the predicted vs true value
plt.figure(figsize=(8,6))
plt.plot(y_test[0:50], label='True Values')
plt.plot(y_test.index[0:50], pred_val, label='Forecast Values')
plt.xlabel('Time Point')
plt.ylabel('Traffic Density')
plt.title(f'50 Step Prediction - Manual\nForecasted Values vs True Values')
plt.legend(loc='upper right')
plt.show()



#Plots the Testing set
model_forecast = model.predict(start=y_train.shape[0], end=y_train.shape[0]+50)

plt.figure(figsize=(8,6))
plt.plot(y_test[0:50], label='True Values')
plt.plot(model_forecast[0:50], label='Statsmodels Forecast Values', alpha=0.9)
plt.plot(y_test.index[0:50], pred_val, label='Manual Forecast Values')
plt.title(f'50 Step Prediction - Statsmodels \nForecasted Values vs True Values')
plt.xlabel('Time Point')
plt.ylabel('Traffic Density')
plt.legend(loc='upper right')
plt.show()

#%%#%%



#One step ahead model - Manual

"""
Do the 1 Step prediction, and plot
"""

#Input variables
steps=y_test.shape[0]


#A series that contains all values needed for the predictions.
#This includes the past values and the predicted values
values=pd.Series(np.zeros((steps+1)))

y_forecast = pd.Series(pd.concat([y_train[-2::] ,  y_test]).values)
#Re-index for easy access
y_forecast.index=[i for i in range(-2, steps)]

#Now, incrementally make predictions    
for i in range(0,steps+1):
    values[i] = 1.374462*y_forecast[i-1] -0.520772*y_forecast[i-2]

one_step = pd.Series(values)

print(f'The Test varience is {y_test.var():0.2f}')
print(f'The Predicted varience is {one_step.var():0.2f}')
print(f'The Varience ratio is {y_test.var()/one_step.var():0.2f}')



#Plot the one step ahead 
plt.figure(figsize=(8,6))
plt.plot(y_test, label='True Values')
plt.plot(predictions_series, label='Statsmodels Forecast Values', alpha=0.9)
plt.plot(y_test.index, one_step[1::], label='Manual Forecast Values', alpha=0.9)
plt.title('ARMA Predicted Parameters Model\n One Step Ahead Testing Set')
plt.xlabel('Time Point')
plt.ylabel('Traffic Density')
plt.legend(loc='lower right')
plt.show()


#%%
"""

ARMA(6,0)

"""



#One step ahead model - Manual

"""
Do the 50 Step prediction, and plot
"""

#Input variables
steps=50
furthest_back=2

#Collector for final set
varience_pred = pd.Series()


#A series that contains all values needed for the predictions.
#This includes the past values and the predicted values
values=pd.Series(np.zeros((steps+furthest_back+1)))

#Re-index for easy access
values.index=[i for i in range(-1*furthest_back, steps+1)]

#Seed the past values
for i in range(-1-furthest_back,0):
    values[i+1] = y_train[i]

#Now, incrementally make predictions    
for i in range(1,steps+1):
    values[i] = 1.374462*values[i-1] -0.520772*values[i-2]

#Include only the predicted values (Exclude the training values)
pred_val=values[furthest_back+1:steps+furthest_back+1]

#Return the predicted values
varience_pred=varience_pred.append(pred_val)

print(f'The Test varience is {y_test.var():0.2f}')
print(f'The Predicted varience is {varience_pred.var():0.2f}')
print(f'The Varience ratio is {y_test.var()/varience_pred.var():0.2f}')


#Create a plot of the predicted vs true value
plt.figure(figsize=(8,6))
plt.plot(y_test[0:50], label='True Values')
plt.plot(y_test.index[0:50], pred_val, label='Forecast Values')
plt.xlabel('Time Point')
plt.ylabel('Traffic Density')
plt.title(f'50 Step Prediction - Manual\nForecasted Values vs True Values')
plt.legend(loc='upper right')
plt.show()



#Plots the Testing set
model_forecast = model.predict(start=y_train.shape[0], end=y_train.shape[0]+50)

plt.figure(figsize=(8,6))
plt.plot(y_test[0:50], label='True Values')
plt.plot(model_forecast[0:50], label='Statsmodels Forecast Values', alpha=0.9)
plt.plot(y_test.index[0:50], pred_val, label='Manual Forecast Values')
plt.title(f'50 Step Prediction - Statsmodels \nForecasted Values vs True Values')
plt.xlabel('Time Point')
plt.ylabel('Traffic Density')
plt.legend(loc='upper right')
plt.show()

#%%



#One step ahead model - Manual

"""
Do the 1 Step prediction, and plot
"""

#Input variables
steps=y_test.shape[0]


#A series that contains all values needed for the predictions.
#This includes the past values and the predicted values
values=pd.Series(np.zeros((steps+1)))

y_forecast = pd.Series(pd.concat([y_train[-2::] ,  y_test]).values)
#Re-index for easy access
y_forecast.index=[i for i in range(-2, steps)]

#Now, incrementally make predictions    
for i in range(0,steps+1):
    values[i] = 1.374462*y_forecast[i-1] -0.520772*y_forecast[i-2]

one_step = pd.Series(values)

print(f'The Test varience is {y_test.var():0.2f}')
print(f'The Predicted varience is {one_step.var():0.2f}')
print(f'The Varience ratio is {y_test.var()/one_step.var():0.2f}')



#Plot the one step ahead 
plt.figure(figsize=(8,6))
plt.plot(y_test, label='True Values')
plt.plot(predictions_series, label='Statsmodels Forecast Values', alpha=0.9)
plt.plot(y_test.index, one_step[1::], label='Manual Forecast Values', alpha=0.9)
plt.title('ARMA Predicted Parameters Model\n One Step Ahead Testing Set')
plt.xlabel('Time (Hourly)')
plt.ylabel('Traffic Volume')
plt.legend(loc='lower right')
plt.show()


