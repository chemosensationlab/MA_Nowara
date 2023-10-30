# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 07:23:05 2023

@author: Chris
"""

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
from numba import jit, cuda 
#@jit(target_backend='cuda') 

from tqdm import tqdm

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

def cal_crosscorr(datax, datay, lag=0):
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

all_path=r'C:\Users\Chris\Desktop\IMARIS traces\20230525\*'
all_path=glob.glob(all_path)

for ind in range(len(all_path)):
    
    try:
        del(trace_1)
        del(trace_2)
        del(b)
    except:
        print('no delete')
        
            

    os.chdir(all_path[ind])
    path=all_path[ind]+'\plot_merged.xlsx'
    #background=r'Y:\File transfer\Michelle_transfer\IMARIS\IMARIS traces\20230531\2020531_Series009_testis\20230531_testis_h_Series009_2_Statistics\20230531_testis_h_Series009_2_Plot_bckgr.xlsx'
    data=pd.read_excel(path)
    data.fillna(0)
    #background=pd.read_excel(background)
    names=data.columns
    names_data=names[1:]
    
    data=np.array(data)
    
    
    
    
    
    # Schleife über die Spalten des Arrays
    #for o in tqdm
    print('Corrections')
    for col in tqdm(range(data.shape[1])):
        # Finden der NaN-Werte in der aktuellen Spalte
        nan_indices = np.isnan(data[:, col])
        
        # Berechnen des Durchschnitts (Mean) der aktuellen Spalte ohne NaN-Werte
        col_mean = np.nanmean(data[:, col])
        
        # Ersetzen der NaN-Werte in der aktuellen Spalte durch den Durchschnitt
        data[nan_indices, col] = col_mean
    
    print('Corrections done')
    
    data=np.delete(data, 0, axis=1)
    
    
    data=pd.DataFrame(data)
    data.columns=names_data
    
    #data_normalized=data
    data_normalized=np.array(data.copy())
    data_smoothed=np.array(data.copy())
    data_average=np.array(data.copy())
    data=np.array(data)
    names_data=names_data[:-1]
    #data_filtered=data
    #data_bin=np.array(data)
    x = np.linspace(1, data_smoothed.shape[0], data_smoothed.shape[0])
    baseline_fitter = Baseline(x_data=x)
    
    
    print('Smoothing...')
    for i in tqdm(range(len(names_data))):
       kernel_size = 11
       kernel = np.ones(kernel_size) / kernel_size
       data_smoothed[:,i]=data_normalized[:,i]-baseline_fitter.modpoly(data[:,i],poly_order=5)[0]
       data_smoothed[:,i] = np.convolve(data_smoothed[:,i], kernel, mode='same')
    print('Smoothing done')
    
    print('Z-Normalization...') 
    for i in tqdm(range(len(names_data))):
        data_normalized[:,i]=stats.zscore(data_smoothed[:,i])
    print('Z-Normalization done')
    #data_normalized=data_normalized.fillna(0)       
    #data_smoothed=data_smoothed.fillna(0)                                   
        
    data_normalized=np.array(data_normalized)
    data_normalized=np.delete(data_normalized,data_normalized.shape[1]-1,axis=1)
    
    #data_filtered=np.array(data_filtered)
    data_bin=np.zeros((data_normalized.shape[0],data_normalized.shape[1]))
    
    
    #data_bin=data_bin[:-1]
    #diff_filtered=np.diff(data_filtered,axis=0)
    
    
    
    data_bin[data_normalized>2.]=1
    
    
    
    print('Binary smoothing')
    for i in tqdm(range(data_bin.shape[1])):
        kernel_size = 35
        kernel = np.ones(kernel_size) / kernel_size
        data_bin[:,i] = np.convolve(data_bin[:,i], kernel, mode='same')
    print('Binary smoothing done')
    data_bin[data_bin!=0]=1
    diff_bin=np.zeros((data_bin.shape[0]-1,len(names_data)))
    
    print('Check for peaks starting in the beginning or the end')
    for col in tqdm(range(data_bin.shape[1])):
        # Finden der NaN-Werte in der aktuellen Spalte
        if data_bin[0, col]==1:
            data_bin[0, col]=0
        if data_bin[-1,col]==1:
            data_bin[-1, col]=0
        diff_bin[:,col]=np.diff(data_bin[:,col])
        
    
    
    peak_start=[]
    peak_end=[]
    print('Creating peak lists')
    for col in tqdm(range(data_bin.shape[1])):
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
    print('Fit the signals')
    for i in tqdm(range(len(peak_start_plus))):
        list_amplitudes=[]
        list_fwhm=[]
        for j in range(len(peak_start_plus[i])):
            x=np.linspace(peak_start_plus[i][j],peak_end_plus[i][j],peak_end_plus[i][j]-peak_start_plus[i][j])
            y=data_smoothed[:,i][peak_start_plus[i][j]:peak_end_plus[i][j]]
            y=np.array(y)
    
    
    
            # Schätzwerte für die Gauß-Fit-Parameter
            A_guess = np.max(y)  # Amplitude schätzen
            mu_guess = x[np.argmax(y)]  # Mittelwert (Position des Maximums) schätzen
            sigma_guess = 2.0  # Standardabweichung schätzen
            
            # Fit durchführen
            try:
                params, covariance = curve_fit(lorentzian, x, y, p0=[A_guess, mu_guess, sigma_guess],maxfev=4000)
            except:
                print('no fit')
            
            # Die ermittelten Parameter ausgeben
            A_fit, mu_fit, sigma_fit = params
            #print(f"Gauß-Fit Parameter: A={A_fit}, mu={mu_fit}, sigma={sigma_fit}")
            
            # Optional: Plot der Daten und des Gauß-Fits
            import matplotlib.pyplot as plt
            
            #plt.figure(figsize=(8, 6))
            #plt.plot(x, y, 'b', label='Daten')
            #x=np.linspace(int(mu_guess)-500,int(mu_guess)+500,1000)
            #if mu_fit>1000:
            #    mu_fit=mu_guess
            #plt.plot(x, lorentzian(x, A_fit, mu_fit, sigma_fit), 'r', label='Lorenz-Fit')
            #plt.xlabel('X-Achse')
            #plt.ylabel('Y-Achse')
            #plt.title(str(max(y)-min(y)))
            #plt.legend()
            #plt.show()
            
            
            try:    
                fwhm=half_max_x(x, lorentzian(x, A_fit, mu_guess, sigma_fit))
                list_amplitudes.append(np.max(y)-np.min(y))
                list_fwhm.append(fwhm[1]-fwhm[0])
                print(fwhm[1]-fwhm[0])
            except:
                print('no fwhm')
                
                    
            
        list_all_amplitudes.append(list_amplitudes)
        list_all_fwhm.append(list_fwhm)
        list_number_peaks.append(len(list_amplitudes))
        
        df=pd.DataFrame(np.concatenate(list_all_amplitudes))
        df.to_excel(r'.\\amplitudes.xlsx', index=False)
        df=pd.DataFrame(np.concatenate(list_all_fwhm))
        df.to_excel(r'.\\fwhm.xlsx', index=False)
        
        df=pd.DataFrame(list_number_peaks)
        df.to_excel(r'.\\number.xlsx', index=False)
        
    
    print('Fit the signals done')   
    list_number_peaks=np.array(list_number_peaks)        
    names_active=names_data[list_number_peaks>1]
    active_indices=np.where(list_number_peaks>1)[0]
    names_active_once=names_data[list_number_peaks>0]
    active_indices_once=np.where(list_number_peaks>0)[0]
    active_diff_bins=diff_bin[:,active_indices_once]
    
    delay=np.zeros((active_diff_bins.shape[1]))
    
    for i in range(active_diff_bins.shape[1]):
        delay[i]=np.where(active_diff_bins[:,i]==1)[0][0]
    
    '''
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
        plt.plot(data[:,i])
        plt.xticks([])
        sns.despine()
        plt.tight_layout()
    '''

    maximum_lag=30
    b=np.zeros((30))
    corr_matrx=np.zeros((len(names_active),len(names_active)))    
    index_max_corr=np.zeros((len(names_active),len(names_active)))
    print('Calculating correlations')
    for it in tqdm(range(len(names_active))):
        for kt in range(len(names_active)):
            trace_1=np.array((data[:,active_indices[it]]))
            trace_2=np.array((data[:,active_indices[kt]]))
            trace_1=pd.DataFrame(trace_1).copy()
            trace_2=pd.DataFrame(trace_2).copy()
            trace_1=trace_1.squeeze()
            trace_2=trace_2.squeeze()
            
            for n in range(maximum_lag):
                p=n
                b[n]=cal_crosscorr(trace_1,trace_2, lag=p)
                corr_matrx[it,kt]=np.nanmax(b)
                
                index_max_corr[it,kt]=(np.where(b==np.nanmax(b))[0][0])
                
    print('Calculating correlations done')
                
    plt.figure()
    plt.imshow(corr_matrx, cmap='rainbow')
    plt.savefig('corr heatmap.svg')
    
    plt.figure()
    plt.imshow(index_max_corr, cmap='rainbow')
    plt.savefig('lag heatmap.svg')
    
    df = pd.DataFrame(corr_matrx)
    df.to_excel(r'.\\correlations.xlsx', index=False)
    
    df = pd.DataFrame(index_max_corr)
    df.to_excel(r'.\\lag.xlsx', index=False)
    
    
    
    def distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    crosscorr=pd.read_excel(r'.\\correlations.xlsx')
    
    path=all_path[ind]+'\pos_merged.xlsx'
    pos=pd.read_excel(path)
    
    x_pos=pos['Track Position X Start']
    y_pos=pos['Track Position Y Start']
    
    x_pos=np.array(x_pos)
    y_pos=np.array(y_pos)
    
    crosscorr=np.array(crosscorr)
    
    
    distance_matrix=np.zeros((len(active_indices),len(active_indices)))
    print('Distance Matrix is in progress...')
    for i_dis in tqdm(range(len(active_indices))):
        for j_dis in range(len(active_indices)):
            distance_matrix[i_dis,j_dis]=distance(x_pos[active_indices[i_dis]],y_pos[active_indices[i_dis]],x_pos[active_indices[j_dis]],y_pos[active_indices[j_dis]])
    print('Distance Matrix done') 
    df = pd.DataFrame(distance_matrix)
    df.to_excel(r'.\\dist.xlsx', index=False)
    crosscorr[distance_matrix>250]=0
    import numpy as np
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    
    # Annahme: x_pos und y_pos sind Ihre Positionsdaten als NumPy-Arrays
    # Sie sollten die Anzahl der Cluster (Zell-Cluster) festlegen.
    num_clusters = 4  # Ändern Sie dies entsprechend Ihren Daten
    
    # Kombinieren Sie x_pos und y_pos zu einer Matrix
    data_matrix = np.column_stack((x_pos, y_pos))
    data_matrix[np.isnan(data_matrix)] = 0
    
    
    wcss = []  # Within-Cluster-Sum-of-Squares
    max_clusters = 10  # Maximale Anzahl von Clustern, die getestet werden
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(data_matrix)
        wcss.append(kmeans.inertia_)
    
    
    
    # Initialisieren und trainieren Sie das K-Means-Modell
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(data_matrix)
    
    # Die Zuordnung jeder Position zu einem Cluster
    cluster_labels = kmeans.labels_
    
    # Die Zentren der Cluster
    cluster_centers = kmeans.cluster_centers_
    
    plt.figure(figsize=(20, 20))
    for i in range(len(delay)):
        plt.plot(x_pos[active_indices_once[i]],y_pos[active_indices_once[i]],'o',c=plt.cm.jet(delay[i]/np.max(delay)))
        plt.yticks([])
        plt.xticks([])
        plt.xlabel('X-Position')
        plt.ylabel('Y-Position')
        sns.despine()
        plt.axis('off')
    plt.savefig('delay.svg')
    
    
    print('Draw figure')    
    for i in tqdm(range(len(active_indices))):
        for j in range(len(active_indices)):
            if i != j:
                if crosscorr[i,j]>0.7 or crosscorr[j,i]>0.7:# Punkte nicht mit sich selbst verbinden
                    plt.plot([x_pos[active_indices[i]], x_pos[active_indices[j]]], [y_pos[active_indices[i]], y_pos[active_indices[j]]], '-',c=plt.cm.jet((crosscorr[i,j]+crosscorr[j,i])/2),lw=5)
    # Erstellen Sie einen Scatterplot, der die Cluster in verschiedenen Farben darstellt
    plt.savefig('network.svg')
    
    plt.figure(figsize=(20, 20))  
    print('Draw figure') 
    
    for cluster_id in range(num_clusters):
        cluster_x = data_matrix[cluster_labels == cluster_id, 0]
        cluster_y = data_matrix[cluster_labels == cluster_id, 1]
        plt.scatter(cluster_x, cluster_y, label=f'Cluster {cluster_id + 1}')
        plt.grid(visible=False)
        plt.yticks([])
        plt.xticks([])
        plt.ylabel([])
        plt.xlabel([])
        plt.axis('off')
        
    for i in tqdm(range(len(active_indices))):
        for j in range(len(active_indices)):
            if i != j:
                if crosscorr[i,j]>0.7 or crosscorr[j,i]>0.7:# Punkte nicht mit sich selbst verbinden
                    plt.plot([x_pos[active_indices[i]], x_pos[active_indices[j]]], [y_pos[active_indices[i]], y_pos[active_indices[j]]], '-',c=plt.cm.jet((index_max_corr[i,j]/30)),lw=5)
    # Erstellen Sie einen Scatterplot, der die Cluster in verschiedenen Farben darstellt
    
        
    plt.savefig('network_lag.svg')
    print('Draw figure done') 
    #plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=100, marker='x', label='Cluster Center')
    plt.xlabel('X-Position')
    plt.ylabel('Y-Position')
    sns.despine()
    #plt.title('Zell-Cluster')
    #plt.legend()
    plt.show()      
