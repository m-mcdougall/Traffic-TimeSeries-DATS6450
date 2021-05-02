# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
import statsmodels
import statsmodels.tsa.holtwinters as ets 
import numpy.linalg as la
from scipy import signal
from scipy.stats import chi2

from statsmodels.regression.linear_model import OLS
#%% 

#========================
#
# Import Previously Written Code
#
#=======================



def ADF_col(df_in, col_in):
    
    adf_calc = adfuller(df_in)
    print(f'\n----- {col_in} -----')
    print("ADF Statistic: %f" %adf_calc[0])
    print('p-value: %f' % adf_calc[1])
    print(f'\n95% Confidence: {["Not Significant","Significant"][int(adf_calc[1]<0.05)]}')
    print(f'99% Confidence: {["Not Significant","Significant"][int(adf_calc[1]<0.01)]}')
    print(f'------------------')


def sub_sequence_stats(df_in, col_in):
    """
    Calculates Mean, Varience and Std for descending sub-sequences of a column of a given dataframe.
    Subsequences start with the first datapoint, then 1st and 2nd, etc until the full dataset is incorperated.
    Returns the stats for all sub-sequences as a dataframe.
    
    df_in: The dataframe to calculate
    col_in: The str name of the column
    """
    
    def stats_calc(df_in, col_in, print_out=False):
        """
        Calculates Mean, Varience and Std for a column of a given dataframe.
        Returns the stats in that order as a tuple.
    
        print_out: Set to True to display the stats in the console
        """
            
        stats_df=df_in.loc[:,col_in]
        
        stats_out=(stats_df.mean(), stats_df.var(), stats_df.std())
        
        if print_out == True:
            print(f'\nThe {col_in} mean is : {stats_out[0] :0.3f}')
            print(f'And the variance is : {stats_out[1] :0.3f}')
            print(f'With standard deviation : {stats_out[2] :0.3f}\n')
    
        return stats_out
       
    
    
    #Collector list
    seq_stats=[]
    
    #Loop through all dataset to make subsections
    for i in range(1,df_in.shape[0]+1):
        seq_stats.append(stats_calc(df_in.head(i), col_in))
    
    #Transform into output dataframe    
    df_seq_stats=pd.DataFrame(seq_stats).rename(columns={0:'Means', 1:'Varience', 2:'Std'})
    
    return df_seq_stats


def plot_sub_sequence_stats(stats_in, col_title):
    """
    Plots the calculated Mean, Varience and Std from the subsequence statistics calculator.
    
    stats_in: The dataframe output by sub_sequence_stats
    col_title: The column used in sub_sequence_stats, to beused as a title string.
    """
    

    stats_in.plot()
    plt.title(f'Statistical Measurements in Subsequences of {col_title}')
    plt.xlabel('Datapoints in Subsequence')
    plt.ylabel('Statistical Measurement')
    plt.legend(loc='upper right')
    
    plt.show()



def auto_corr_cal(array_in, k):
    '''
    Calculates the autocorrelation of a given array at time lag k
    
    Returns the autocorrelation at k.
    '''
    
    if k==0:
        return 1
    
    elif k >= len(array_in):
        raise ValueError ('K exceeds maximum value.')
        
    else:
        #Calculate the utocorrelation
        
        t=k+1-1#-1 for 0 indexing   
        
        #Shift array by t
        y_t = array_in[t::]
        
        #Array t-k must always start at 1, since t=k+1, t-k will always be 1. (then starts at 0 b/c 0 index)
        #Cuts off at -k to make it the same length as y_t for broadcasting
        y_t_minusk = array_in[0:-k] 
        
        #Calculate the mean of all samples
        y_mean = array_in.mean()
        
        #Subtract mean fom all elements of the array
        y_t = y_t-y_mean
        y_t_minusk = y_t_minusk-y_mean
        
        #Break the operation into two steps for legibility
        numerator = sum( y_t*y_t_minusk)
        denom = sum((array_in - y_mean)**2)
        
        return numerator/denom
    


def symmetric_array(array_in):
    """
    Creates a symmetrical array by reversing and concatinating your array
    """
    
    if type(array_in)== np.ndarray:
        array_in=list(array_in)
    
    array_in_full = array_in[::-1]
    array_in_full=array_in_full+array_in[1::]
    
    return np.array(array_in_full)



def run_auto_corr(array_in, lags, symmetrical=False):
    """
    Takes an array of values and runs an autocorraltion up to the desired number of lags
    If symmetrical == True, will return a symmetircal array centered on Rk=1
    """
    if type(array_in)!=np.ndarray:
        raise ValueError("Please enter array_in as an array")
    
    array_in = array_in[~np.isnan(array_in)]
    
    #Add a lag, to account for lag=0
    lags+=1 

    collect=[]
    for i in range(lags):
        collect.append(auto_corr_cal(array_in, i))

    if symmetrical == True:
        collect=symmetric_array(collect)
        
    else:
        collect=np.array(collect)

    return collect



def plot_autocorrelation_simple(array_in, title_str, original_array):
    """
    Plots a simple autocorrelation stem plot
    """
    
    #Check is symmetrical
    if array_in[0] == array_in[-1] and array_in[1] == array_in[-2]:
        
        xrange = int((len(array_in)-1)/2)
        xrange = range(-xrange, xrange+1)

    else:
        xrange =range(len(array_in))
    
    #Generate the figure
    fig = plt.figure()
    plt.stem(xrange, array_in, use_line_collection=True)
    
    m=1.96/np.sqrt(len(original_array))
    plt.axhspan(-m,m,alpha = .1, color = 'black')
    
    plt.xlabel('Lags')
    plt.ylabel('Correlation')
    plt.title(title_str)
    plt.show()




def createGPACTable(auto_corr_in, j_val=12, k_val=12):
    """
    Creates the full GPAC table, to feed into the plotting function
    """
    
    if auto_corr_in[0]== auto_corr_in[-1]:
        response = input('This requires a non-symmetrical Autocorrelation. Is this non-symetrical?')
        
        if response.lower() not in ['yes','y','go']:
            raise ValueError('Please enter a non-symmetrical array.')
    
    j_val+=1
    k_val+=1        
            
        
    big_out=np.zeros([j_val, k_val])
    
    for k in range(1,k_val):
        for j in range(0,j_val):
            
            full_array=[]
            
            #Generate the top row
            ry_array=[i for i in range((j-k+1), j+1)] #Generates the array for the top row of the denominator, but backwards
            ry_array=ry_array[::-1] #Reverse to proper order
            ry_array=np.array(ry_array)
            
            #Generate the rest based off that row
            for i in range(0,k):
                full_array.append(ry_array+i)
            
            #Concat to a single array
            full_array=np.array(full_array)
            
            
            
            #Make the nuerator array
            numer_array=full_array.copy()
            
            #The fiinal column of the numerator is different. Generate it here
            numer_col=[i for i in range(j+1, j+k+1)]
            
            #Replace the final col
            numer_array[:,-1]=numer_col
            
            
            #Take the absolute values  - all negative Ry are equivalent to Positive Ry
            numer_array=abs(numer_array)*1.0
            denom_array=abs(full_array)*1.0
            
            #Get the respective numbers from the autocor array
            
            
            numer_array2=numer_array.copy()
            
            for i in range(numer_array2.shape[0]):
                for n in range(numer_array2.shape[1]):
                    numer_array2[i,n] = auto_corr_in[int(numer_array2[i,n])]
                   
                    
            
            denom_array2=denom_array.copy()
            
            for i in range(denom_array2.shape[0]):
                for n in range(denom_array2.shape[1]):
                    denom_array2[i,n] = auto_corr_in[int(denom_array2[i,n])]
                    
                    
            
            val=np.linalg.det(numer_array2)/np.linalg.det(denom_array2)
            
            big_out[j,k]=val
            big_out_display=pd.DataFrame(big_out)
            
    big_out_display=pd.DataFrame(big_out)

    return big_out_display.iloc[:, 1:]
    
          

def plotGPAC(gpac_table,equation_string='', decimals=1):
    """
    Plots the output of the createGPACTable as a heatmap plot
    Can change the number of decimal places shown in the annotation using the decimals parameter.
    """    
    num_format='0.'+str(decimals)+'f'
    if equation_string != '':
        title_str = "GPAC Table\n"+ equation_string
    else:
        title_str = "GPAC Table"
    
    fig, ax = plt.subplots(figsize=[9,7])
    sns.heatmap(gpac_table, vmin=-1, vmax=1, cmap='PRGn', annot=True, fmt=num_format, ax=ax)
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = True, bottom=True, top = True, labeltop=True)
    plt.yticks(rotation=0)
    plt.xlabel('k  (i.e. na)', fontsize=15)
    plt.ylabel('j  (i.e. nb)', fontsize=15)
    plt.title(title_str, fontsize=15)




def createGPAC(system_in, equation_string='', j_val=12, k_val=12):
    """
    Creates the GPAC table, including running the autocorrelation and plotting
    Input data as an non-autocorrelated array
    """
    autocor_system = run_auto_corr(system_in, lags=40, symmetrical=False)
    gpac_table_plot = createGPACTable(autocor_system, j_val, k_val)
    plotGPAC(gpac_table_plot, equation_string)
    



def statstoolsACF_PACF(system_in, lags=20, title_str=''):
    """
    Uses the Statstools packages to plot ACF and PACF
    """
    ## Plot the ACF and PACF
    acf = sm.tsa.stattools.acf(system_in, nlags=lags, fft=False)
    pacf = sm.tsa.stattools.pacf(system_in, nlags=lags)
    
    plt.figure()
    plt.subplot(211)
    plot_acf(system_in, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(system_in, ax=plt.gca(), lags=lags)
    plt.xlabel('Lags')
    plt.suptitle(f'ACF and PACF:\n {title_str}', y=1.08)
    plt.tight_layout()
    plt.show()


def plot_prediction_method_axis(train_in, test_in, pred_in, error2, method_str):
    """
    Plots the training testing and forecasted data given a prediction method.
    """
    fig = plt.figure(figsize=[10,6])
    ax = fig.add_subplot(1,1,1)
    
    pred_diff = len(test_in) - len(pred_in)
    
    ax.plot(train_in.index, train_in,  linewidth=1, label= "Training Set")
    ax.plot(test_in.index, test_in,  linewidth=1, label= "Testing Set")
    ax.plot(test_in.index[pred_diff::], pred_in,  linewidth=1, label= "Forecast", alpha=0.8)
    ax.set_xlabel('Time (Hourly)')
    ax.set_ylabel('Traffic Volume')
    ax.set_title(f'Traffic Predictions using {method_str}\n MSE: {error2.mean() :0.2f}', fontsize=14)
    ax.legend()
    plt.show()
    
    return 



def average_test(train_in, test_in):
    """
    Takes a training set and a testing set, and generates a testing forecast using the average method
    returns the predicted values, error and mse
    
    train_in: a training array
    test_in: a testing array
    """
    
    if type(train_in)==list:
        train_in=np.array(train_in)

    if type(test_in)==list:
        train_in=np.array(test_in)
    
    #Take mean of training set, this will be the value for all testing points    
    train_mean = np.mean(train_in)
    pred_y=np.ones([len(test_in)])*train_mean
    
    #Calculate error
    error = test_in - pred_y
    error_2 = error**2
    
    return pred_y, error, error_2



def naive_test(train_in, test_in):
    """
    Takes a training set and a testing set, and generates a testing forecast using the naive method
    returns the predicted values, error and mse
    
    train_in: a training array
    test_in: a testing array
    """
    
    if type(train_in)==list:
        train_in=np.array(train_in)

    if type(test_in)==list:
        test_in=np.array(test_in)
    
    #Take mean of training set, this will be the value for all testing points    
    pred_y=np.ones([len(test_in)])*train_in[-1]
    
    #Calculate error
    error = test_in - pred_y
    error_2 = error**2
    
    return pred_y, error, error_2



def drift_test(train_in, test_in):
    """
    Takes a training set and a testing set, and generates a testing forecast using the drift method
    returns the predicted values, error and mse
    
    train_in: a training array
    test_in: a testing array
    """
    
    if type(train_in)==list:
        train_in=np.array(train_in)

    if type(test_in)==list:
        test_in=np.array(test_in)
        
    pred_y=[]
    
    #Loop through the testing values
    for i in range(0,len(test_in)):
        
        val = train_in[-1] + (i+1)*((train_in[-1]-train_in[0]) / (len(train_in)-1))
        pred_y.append(val)
    
    #Calculate error
    error = test_in - pred_y
    error_2 = error**2
    
    return pred_y, error, error_2

def ses_train(train_in, alpha=0.5):
    """
    Takes a training set and generates a training forecast using the ses method
    returns the predicted vales, error and mse
    
    train_in: a training array
    """
    
    if type(train_in)==list:
        train_in=np.array(train_in)
        
    #Capture predicted values -pre-load with 1st value
    pred_y=[train_in[0]]
    
    #Loop through and get the predicted values
    for i in range(1,len(train_in)):
        
        val = alpha*train_in[i] + (1-alpha)*pred_y[-1]
        pred_y.append(val)
    

    #Calculate error
    error = np.array(train_in[1::]) - np.array(pred_y[0:-1])
    error_2 = error**2
    
    return pred_y, error, error_2


def ses_test(train_in, test_in, alpha=0.5):
    """
    Takes a training set and a testing set, and generates a testing forecast using the ses method
    returns the predicted values, error and mse
    
    train_in: a training array
    test_in: a testing array
    """
    
    if type(train_in)==list:
        train_in=np.array(train_in)

    if type(test_in)==list:
        test_in=np.array(test_in)
    
    #Get the prediction for the final value in the training set
    final_pred = ses_train(train_in, alpha=0.5)[0][-1] 
    
    val = alpha * test_in[-1] + (1-alpha)*final_pred

    #Take mean of training set, this will be the value for all testing points    
    pred_y=np.ones([len(test_in)])*val
    
    #Calculate error
    error = test_in - pred_y
    error_2 = error**2
    
    return pred_y, error, error_2


def forecast_error_varience_calc(error_in, x_in):
    """
    Input the error calculated and the x_test used to make the predicition
    
    Returns estimated varience
    """
    
    #Extract the T and k values from the dimensions of the input x data
    T = x_in.shape[0] #Number of samples
    k = x_in.shape[1] #Number of features
    
    #Calculate the varience
    varience = np.sqrt( (1/(T-k-1)) * np.sum(error_in**2))
    return varience

    
def calc_Q_Score(residuals, train_in, lags=5, print_out=False):
    """
    Residuals = Error term from the training set (NOT testing)
    """
    
    num_samples=len(train_in)
    auto_corr =[]
    
    residuals = residuals[~np.isnan(residuals)]
    
    for i in range(lags):
        auto_corr.append(auto_corr_cal(residuals, i))
    
    summed = np.sum(np.array(auto_corr)[1:]**2)
    q_score = num_samples * summed
    
    if print_out !=False:
        print(f' The Q Score is {q_score: 0.3f}\n')
    return q_score




def createGPACTable(auto_corr_in, j_val=12, k_val=12):
    """
    Creates the full GPAC table, to feed into the plotting function
    """
    
    if auto_corr_in[0]== auto_corr_in[-1]:
        response = input('This requires a non-symmetrical Autocorrelation. Is this non-symetrical?')
        
        if response.lower() not in ['yes','y','go']:
            raise ValueError('Please enter a non-symmetrical array.')
    
    j_val+=1
    k_val+=1        
            
        
    big_out=np.zeros([j_val, k_val])
    
    for k in range(1,k_val):
        for j in range(0,j_val):
            
            full_array=[]
            
            #Generate the top row
            ry_array=[i for i in range((j-k+1), j+1)] #Generates the array for the top row of the denominator, but backwards
            ry_array=ry_array[::-1] #Reverse to proper order
            ry_array=np.array(ry_array)
            
            #Generate the rest based off that row
            for i in range(0,k):
                full_array.append(ry_array+i)
            
            #Concat to a single array
            full_array=np.array(full_array)
            
            
            
            #Make the nuerator array
            numer_array=full_array.copy()
            
            #The fiinal column of the numerator is different. Generate it here
            numer_col=[i for i in range(j+1, j+k+1)]
            
            #Replace the final col
            numer_array[:,-1]=numer_col
            
            
            #Take the absolute values  - all negative Ry are equivalent to Positive Ry
            numer_array=abs(numer_array)*1.0
            denom_array=abs(full_array)*1.0
            
            #Get the respective numbers from the autocor array
            
            
            numer_array2=numer_array.copy()
            
            for i in range(numer_array2.shape[0]):
                for n in range(numer_array2.shape[1]):
                    numer_array2[i,n] = auto_corr_in[int(numer_array2[i,n])]
                   
                    
            
            denom_array2=denom_array.copy()
            
            for i in range(denom_array2.shape[0]):
                for n in range(denom_array2.shape[1]):
                    denom_array2[i,n] = auto_corr_in[int(denom_array2[i,n])]
                    
                    
            
            val=np.linalg.det(numer_array2)/np.linalg.det(denom_array2)
            
            big_out[j,k]=val
            big_out_display=pd.DataFrame(big_out)
            
    big_out_display=pd.DataFrame(big_out)

    return big_out_display.iloc[:, 1:]
    
          

def plotGPAC(gpac_table,equation_string='', decimals=1):
    """
    Plots the output of the createGPACTable as a heatmap plot
    Can change the number of decimal places shown in the annotation using the decimals parameter.
    """    
    num_format='0.'+str(decimals)+'f'
    if equation_string != '':
        title_str = "GPAC Table\n"+ equation_string
    else:
        title_str = "GPAC Table"
    
    fig, ax = plt.subplots(figsize=[9,7])
    sns.heatmap(gpac_table, vmin=-1, vmax=1, cmap='PRGn', annot=True, fmt=num_format, ax=ax)
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = True, bottom=True, top = True, labeltop=True)
    plt.yticks(rotation=0)
    plt.xlabel('k  (i.e. na)', fontsize=15)
    plt.ylabel('j  (i.e. nb)', fontsize=15)
    plt.title(title_str, fontsize=15)




def createGPAC(system_in, equation_string='', j_val=12, k_val=12):
    """
    Creates the GPAC table, including running the autocorrelation and plotting
    Input data as an non-autocorrelated array
    """
    autocor_system = run_auto_corr(system_in, lags=40, symmetrical=False)
    gpac_table_plot = createGPACTable(autocor_system, j_val, k_val)
    plotGPAC(gpac_table_plot, equation_string)
    



def statstoolsACF_PACF(system_in, lags=20, title_str=''):
    """
    Uses the Statstools packages to plot ACF and PACF
    """
    ## Plot the ACF and PACF
    acf = sm.tsa.stattools.acf(system_in, nlags=lags, fft=False)
    pacf = sm.tsa.stattools.pacf(system_in, nlags=lags)
    
    plt.figure()
    plt.subplot(211)
    plot_acf(system_in, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(system_in, ax=plt.gca(), lags=lags)
    plt.xlabel('Lags')
    plt.suptitle(f'ACF and PACF:\n {title_str}', y=1.08)
    plt.tight_layout()
    plt.show()

def diff_seasonal_calc(df_in, interval=1):
    '''
    Seasonal differencing - so it omits the intermediate values not in the seasonal increment.
    '''
    diff = []
    for i in range(interval, len(df_in)):
      value = df_in[i] - df_in[i - interval]
      diff.append(value)

    diff=pd.DataFrame(diff)
    return diff

def zero_pole_print(poly_in):
    """
    Prints the equation string for zeros or poles
    """
    collect=[]
    for i in poly_in:
        if i<0:
            collect.append(str(f"(1+{i*-1:0.4f})"))
        else:
            collect.append(str(f"(1-{i:0.4f})"))
    str_out=''
    for i in collect:
        str_out+=i
    print(str_out)
    
def create_samples(n_samples, wn_mean=0, wn_var=1):
    """
    Creates a normally distrubuted random dataset
    Import number of samples, mean and std.
    """
    
    #Set variables
    wn_std = np.sqrt(wn_var)
    
    #Guarentee same white noise for replicability
    np.random.seed(42)
    
    #Create the sample
    e = np.random.normal(wn_mean, wn_std, size=n_samples)
    
    return e

