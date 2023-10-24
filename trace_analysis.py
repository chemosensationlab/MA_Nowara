# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:10:42 2023

@author: wiesbrock
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.stats as stats
import seaborn as sns
from scipy.signal import argrelextrema
from tqdm import tqdm
from read_roi import read_roi_file
import math
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import pandas as pd
from scipy.signal import butter, sosfreqz, sosfilt

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

from pybaselines import Baseline
from pybaselines.utils import gaussian

def first_non_zero_elem(arr):
    for element in arr:
        if element != 0:
            return element
    return None

def set_zero(arr):
    for i, element in enumerate(arr):
        if element != 0:
            if element == -1:
                arr[i] = 0
            break
        
def substract_arrays(liste1, liste2):
    ergebnis = []
    
    # Stellen Sie sicher, dass beide Listen die gleiche Länge haben
    if len(liste1) != len(liste2):
        raise ValueError("Die Listen haben unterschiedliche Längen.")
    
    for arr1, arr2 in zip(liste1, liste2):
        if len(arr1) != len(arr2):
            raise ValueError("Die Arrays in den Listen haben unterschiedliche Längen.")
        
        differenz = [a - b for a, b in zip(arr1, arr2)]
        ergebnis.append(differenz)
    
    return ergebnis

def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]

# Lorenz-Funktion definieren
def lorentzian(x, A, x0, gamma):
    try:
        result = A / (1 + ((x - x0) / gamma) ** 2)
        return result
    except Exception as e:
        print("Fehler beim Berechnen des Lorentzian:", str(e))
        return None

def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

# make some fake data


# Hochpassfilter-Parameter
cutoff_frequency = 0.01  # Cut-off-Frequenz in Hz
sampling_rate = 2.7  # Samplingrate in Hz
order = 1  # Filterordnung

# Berechnen der Filterkoeffizienten
nyquist = 0.5 * sampling_rate
high = cutoff_frequency / nyquist
sos = butter(order, high, btype='low', analog=False, output='sos')
# Berechnen der Frequenzantwort des Filters
frequencies, response = sosfreqz(sos, worN=2000, fs=sampling_rate)



path=r'Y:\File transfer\Michelle_transfer\IMARIS\IMARIS traces\20230612\20230612_testis_ATP_3.1\ATP on testis_c_ATP3.1_0_Statistics\ATP on testis_c_ATP3.1_0_Plot.csv'
#background=r'Y:\File transfer\Michelle_transfer\IMARIS\IMARIS traces\20230531\2020531_Series009_testis\20230531_testis_h_Series009_2_Statistics\20230531_testis_h_Series009_2_Plot_bckgr.xlsx'
data=pd.read_csv(path,skiprows=(0,1,2))
data.fillna(0)
#background=pd.read_excel(background)
names=data.columns
names_data=names[1:]

data=np.array(data)





# Schleife über die Spalten des Arrays
for col in range(data.shape[1]):
    # Finden der NaN-Werte in der aktuellen Spalte
    nan_indices = np.isnan(data[:, col])
    
    # Berechnen des Durchschnitts (Mean) der aktuellen Spalte ohne NaN-Werte
    col_mean = np.nanmean(data[:, col])
    
    # Ersetzen der NaN-Werte in der aktuellen Spalte durch den Durchschnitt
    data[nan_indices, col] = col_mean


data=np.delete(data, 0, axis=1)


data=pd.DataFrame(data)
data.columns=names_data

#data_normalized=data
data_normalized=data.copy()
data_smoothed=data.copy()
data_average=data.copy()
names_data=names_data[:-1]
#data_filtered=data
#data_bin=np.array(data)
x = np.linspace(1, data_smoothed.shape[0], data_smoothed.shape[0])
baseline_fitter = Baseline(x_data=x)

for i in range(len(names_data)):
   kernel_size = 11
   kernel = np.ones(kernel_size) / kernel_size
   data_smoothed[names_data[i]]=data_normalized[names_data[i]]-baseline_fitter.modpoly(data[names_data[i]],poly_order=5)[0]
   data_smoothed[names_data[i]] = np.convolve(data_smoothed[names_data[i]], kernel, mode='same')


 
for i in range(len(names_data)):
    data_normalized[names_data[i]]=stats.zscore(data_smoothed[names_data[i]])

                                           
    
data_normalized=np.array(data_normalized)
data_normalized=np.delete(data_normalized,data_normalized.shape[1]-1,axis=1)

#data_filtered=np.array(data_filtered)
data_bin=np.zeros((data_normalized.shape[0],data_normalized.shape[1]))


#data_bin=data_bin[:-1]
#diff_filtered=np.diff(data_filtered,axis=0)



data_bin[data_normalized>2.]=1




for i in range(data_bin.shape[1]):
    kernel_size = 35
    kernel = np.ones(kernel_size) / kernel_size
    data_bin[:,i] = np.convolve(data_bin[:,i], kernel, mode='same')

data_bin[data_bin!=0]=1
diff_bin=np.zeros((data_bin.shape[0]-1,len(names_data)))


for col in range(data_bin.shape[1]):
    # Finden der NaN-Werte in der aktuellen Spalte
    if data_bin[0, col]==1:
        data_bin[0, col]=0
    diff_bin[:,col]=np.diff(data_bin[:,col])

peak_start=[]
peak_end=[]
for col in range(data_bin.shape[1]):
    # Finden der NaN-Werte in der aktuellen Spalte
    peak_start.append(np.where(diff_bin[:,col]==1))
    peak_end.append(np.where(diff_bin[:,col]==-1))
    


peak_start_plus = []
peak_end_plus = []

# Schleife, um Werte von 0 zu entfernen und die entsprechenden Werte in der zweiten Liste zu entfernen
for start_array, end_array in zip(peak_start, peak_end):
    nicht_null_mask = np.where(start_array[0] != 0)  # Maske für Nicht-Null-Werte in peak_start
    neue_start_array = start_array[0][nicht_null_mask]
    neue_end_array = end_array[0][nicht_null_mask]
    peak_start_plus.append(neue_start_array)
    peak_end_plus.append(neue_end_array)
    
    
list_all_amplitudes=[]
list_all_fwhm=[]
list_number_peaks=[]
for i in  range(len(peak_start_plus)):
    list_amplitudes=[]
    list_fwhm=[]
    for j in range(len(peak_start_plus[i])):
        x=np.linspace(peak_start_plus[i][j],peak_end_plus[i][j],peak_end_plus[i][j]-peak_start_plus[i][j])
        y=data_smoothed[names_data[i]][peak_start_plus[i][j]:peak_end_plus[i][j]]
        y=np.array(y)



        # Schätzwerte für die Gauß-Fit-Parameter
        A_guess = np.max(y)  # Amplitude schätzen
        mu_guess = x[np.argmax(y)]  # Mittelwert (Position des Maximums) schätzen
        sigma_guess = 2.0  # Standardabweichung schätzen
        
        # Fit durchführen
        params, covariance = curve_fit(lorentzian, x, y, p0=[A_guess, mu_guess, sigma_guess],maxfev=4000)
        
        # Die ermittelten Parameter ausgeben
        A_fit, mu_fit, sigma_fit = params
        print(f"Gauß-Fit Parameter: A={A_fit}, mu={mu_fit}, sigma={sigma_fit}")
        
        # Optional: Plot der Daten und des Gauß-Fits
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, 'b', label='Daten')
        x=np.linspace(int(mu_fit)-500,int(mu_fit)+500,1000)
        plt.plot(x, lorentzian(x, A_fit, mu_fit, sigma_fit), 'r', label='Lorenz-Fit')
        plt.xlabel('X-Achse')
        plt.ylabel('Y-Achse')
        plt.title(str(max(y)-min(y)))
        plt.legend()
        plt.show()
        
        
        fwhm=half_max_x(x, lorentzian(x, A_fit, mu_fit, sigma_fit))
        list_amplitudes.append(np.max(y)-np.min(y))
        list_fwhm.append(fwhm[1]-fwhm[0])
        print(fwhm[1]-fwhm[0])
        
    list_all_amplitudes.append(list_amplitudes)
    list_all_fwhm.append(list_fwhm)
    list_number_peaks.append(len(list_amplitudes))
    
        

for i in range(len(names_data)-1):
    plt.figure()
    
    plt.subplot(313)
    plt.title('Peaks')
    plt.plot(data_bin[:,i])
    sns.despine()
    plt.subplot(312)
    plt.title('Corrected')
    plt.plot(data_normalized[:,i])
    sns.despine()
    plt.xticks([])
    plt.subplot(311)
    plt.title('raw')
    plt.plot(data[names_data[i]])
    plt.xticks([])
    sns.despine()
    plt.tight_layout()
    
maximum_lag=60
b=np.zeros((60))
corr_matrx=np.zeros((len(names_data),len(names_data)))    
index_max_corr=np.zeros((len(names_data),len(names_data)))
for i in range(len(names_data)):
    for k in range(len(names_data)):
        trace_1=(data_smoothed[names_data[i]])
        trace_2=(data_smoothed[names_data[k]])
        trace_1=pd.DataFrame(trace_1)
        trace_2=pd.DataFrame(trace_2)
        trace_1=trace_1.squeeze()
        trace_2=trace_2.squeeze()
        
        for n in range(maximum_lag):
            p=n
            b[n]=crosscorr(trace_1,trace_2, lag=p)
            corr_matrx[i,k]=np.nanmax(b)
            
            index_max_corr[i,k]=(np.where(b==np.nanmax(b))[0][0])
            
plt.figure()
plt.imshow(corr_matrx, cmap='rainbow')

df = pd.DataFrame(corr_matrx)
df.to_excel(r'.\\correlations.xlsx', index=False)
    


    
    
    
    
    
    
    
