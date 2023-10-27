# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:41:03 2023

@author: wiesbrock
"""

import glob
import numpy as np
import pandas as pd
import os

path=r'C:\Users\wiesbrock\Desktop\IMARIS traces\20230522\20230522_glass_spont_S001\*'
files=glob.glob(path)

list_plot=[]
list_pos=[]

for i in files:
    if "Statistics" in i and "Pos" not in i:
        list_plot.append(i)
        
for i in files:
    if "Pos" in i:
        list_pos.append(i)
        
list_plot=np.array(list_plot)
list_pos=np.array(list_pos)

files_plot=[]
files_pos=[]
for i in list_plot:
    files_plot.append(glob.glob(i+'\*'))
    
for i in list_pos:
    files_pos.append(glob.glob(i+'\*'))
    
#files_plot=np.array(files_plot)
#files_pos=np.array(files_pos)

# Name der neuen Excel-Datei, in die Sie die Spalten übertragen möchten
ziel_datei = path[:-1]+'\\plot_merged.xlsx'

# DataFrame erstellen, um die Daten aus den verschiedenen Dateien zu speichern
gesamt_df = pd.DataFrame()

for i in files_plot:
    df = pd.read_csv(i[0],skiprows=(0,1,2))  # Excel-Datei in ein DataFrame laden
    gesamt_df = pd.concat([gesamt_df, df], axis=1)

gesamt_df = gesamt_df.dropna(axis=1, how='all')
gesamt_df = gesamt_df.drop(columns=['Time [s]'])  
gesamt_df.to_excel(ziel_datei, index=False)

ziel_datei = path[:-1]+'\\pos_merged.xlsx'

# DataFrame erstellen, um die Daten aus den verschiedenen Dateien zu speichern
pos_df = pd.DataFrame()

for i in files_pos:
    df = pd.read_csv(i[0],skiprows=(0,1,2))  # Excel-Datei in ein DataFrame laden
    pos_df = pd.concat([pos_df, df], axis=0)

#pos_df = gesamt_df.drop(columns=['Time [s]'])  
pos_df.to_excel(ziel_datei, index=False)
    
