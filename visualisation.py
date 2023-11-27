# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:42:12 2023

@author: wiesbrock
"""

import glob
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pylab as plt
import scipy.stats as stats

ending='number_list.csv'
title='Number'
label='Number of Signals [%]'

blues = sns.color_palette("Blues_r", n_colors=4, desat=1)
oranges= sns.color_palette("Oranges_r", n_colors=4, desat=1)

path_glass_s=r'C:\Users\Chris\Desktop\ATP\IMARIS traces\Glass_s\\'+ending
path_testis_s=r'C:\Users\Chris\Desktop\ATP\IMARIS traces\Testis_s\\'+ending
path_atp_1=r'C:\Users\Chris\Desktop\ATP\IMARIS traces\ATP 1\\'+ending
path_atp_3=r'C:\Users\Chris\Desktop\ATP\IMARIS traces\ATP 3\\'+ending
path_atp_10=r'C:\Users\Chris\Desktop\ATP\IMARIS traces\ATP 10\\'+ending
path_atp_1_testis=r'C:\Users\Chris\Desktop\ATP\IMARIS traces\ATP 1_testis\\'+ending
path_atp_3_testis=r'C:\Users\Chris\Desktop\ATP\IMARIS traces\ATP 3_testis\\'+ending
path_atp_10_testis=r'C:\Users\Chris\Desktop\ATP\IMARIS traces\ATP 10_testis\\'+ending

glass_s=pd.read_csv(path_glass_s)
testis_s=pd.read_csv(path_testis_s)
atp_1=pd.read_csv(path_atp_1)
atp_3=pd.read_csv(path_atp_3)
atp_10=pd.read_csv(path_atp_10)
atp_1_testis=pd.read_csv(path_atp_1_testis)
atp_3_testis=pd.read_csv(path_atp_3_testis)
atp_10_testis=pd.read_csv(path_atp_10_testis)

glass_s=np.concatenate(np.array(glass_s))
testis_s=np.concatenate(np.array(testis_s))
atp_1=np.concatenate(np.array(atp_1))
atp_3=np.concatenate(np.array(atp_3))
atp_10=np.concatenate(np.array(atp_10))
atp_1_testis=np.concatenate(np.array(atp_1_testis))
atp_3_testis=np.concatenate(np.array(atp_3_testis))
atp_10_testis=np.concatenate(np.array(atp_10_testis))

# Create a boolean mask to identify NaN values
nan_mask = np.isnan(glass_s)

# Use the mask to filter out NaN values
glass_s = glass_s[~nan_mask]

# Create a boolean mask to identify NaN values
nan_mask = np.isnan(testis_s)

# Use the mask to filter out NaN values
testis_s = testis_s[~nan_mask]

# Create a boolean mask to identify NaN values
nan_mask = np.isnan(atp_1)

# Use the mask to filter out NaN values
atp_1 = atp_1[~nan_mask]

# Create a boolean mask to identify NaN values
nan_mask = np.isnan(atp_3)

# Use the mask to filter out NaN values
atp_3 = atp_3[~nan_mask]

# Create a boolean mask to identify NaN values
nan_mask = np.isnan(atp_10)

# Use the mask to filter out NaN values
atp_10 = atp_10[~nan_mask]

# Create a boolean mask to identify NaN values
nan_mask = np.isnan(atp_1_testis)

# Use the mask to filter out NaN values
atp_1_testis = atp_1_testis[~nan_mask]

# Create a boolean mask to identify NaN values
nan_mask = np.isnan(atp_3_testis)

# Use the mask to filter out NaN values
atp_3_testis = atp_3_testis[~nan_mask]

# Create a boolean mask to identify NaN values
nan_mask = np.isnan(atp_10_testis)

# Use the mask to filter out NaN values
atp_10_testis = atp_10_testis[~nan_mask]

print(stats.kruskal(glass_s,testis_s,atp_1,atp_3,atp_10,atp_1_testis,atp_3_testis,atp_10_testis))

#data=glass_s,testis_s,atp_1,atp_3,atp_10,atp_1_testis,atp_3_testis,atp_10_testis

#glass_s=glass_s[glass_s<np.mean(glass_s)+np.std(glass_s)]
#testis_s=testis_s[testis_s<np.mean(testis_s)+np.std(testis_s)]

#data=glass_s,testis_s

'''
###amplitude und fwhm###
atp_1_testis=atp_1_testis[atp_1_testis<np.mean(atp_1_testis)+2*np.std(atp_1_testis)]
atp_3_testis=atp_3_testis[atp_3_testis<np.mean(atp_3_testis)+2*np.std(atp_3_testis)]
atp_10_testis=atp_10_testis[atp_10_testis<np.mean(atp_10_testis)+2*np.std(atp_10_testis)]

atp_1=atp_1[atp_1<np.mean(atp_1_testis)+2*np.std(atp_1_testis)]
atp_3=atp_3[atp_3<np.mean(atp_3)+2*np.std(atp_3)]
atp_10=atp_10[atp_10<np.mean(atp_10)+2*np.std(atp_10)]

glass_s=glass_s[glass_s<np.mean(glass_s)+2*np.std(glass_s)]
testis_s=testis_s[testis_s<np.mean(testis_s)+2*np.std(testis_s)]
'''

'''
###correlation###
atp_1_testis=atp_1_testis[atp_1_testis<1]
atp_3_testis=atp_3_testis[atp_3_testis<1]
atp_10_testis=atp_10_testis[atp_10_testis<1]

atp_1=atp_1[atp_1<1]
atp_3=atp_3[atp_3<1]
atp_10=atp_10[atp_10<1]

glass_s=glass_s[glass_s<1]
testis_s=testis_s[testis_s<1]
'''

data=atp_1_testis,atp_3_testis,atp_10_testis
plt.figure(dpi=300)
ax=sns.violinplot(data=data, cut=0)

print('ATP_1_testis '+str(np.mean(atp_1_testis))+"_+-_"+str(np.std(atp_1_testis)))
print('ATP_3_testis '+str(np.mean(atp_3_testis))+"_+-_"+str(np.std(atp_3_testis)))
print('ATP_10_testis '+str(np.mean(atp_10_testis))+"_+-_"+str(np.std(atp_10_testis)))

sns.despine()
#plt.ylim(0.0,100)
plt.title('ATP testis')
plt.ylabel(label)

ax.set_xticklabels([u'ATP 1 \u03bcMol',u'ATP 3 \u03bcMol','ATP 10 \u03bcMol'])

plt.savefig(str(title)+'_ATP_testis.jpg')
plt.savefig(str(title)+'_ATP_testis.svg')

data=atp_1,atp_3,atp_10
plt.figure(dpi=300)
ax=sns.violinplot(data=data, cut=0)

print('ATP_1 '+str(np.mean(atp_1))+"_+-_"+str(np.std(atp_1)))
print('ATP_3 '+str(np.mean(atp_3))+"_+-_"+str(np.std(atp_3)))
print('ATP_10 '+str(np.mean(atp_10))+"_+-_"+str(np.std(atp_10)))

sns.despine()
#plt.ylim(0.0,100)
plt.title('ATP glass')
plt.ylabel(label)

ax.set_xticklabels([u'ATP 1 \u03bcMol',u'ATP 3 \u03bcMol','ATP 10 \u03bcMol'])

plt.savefig(str(title)+'_ATP_glass.jpg')
plt.savefig(str(title)+'_ATP_glass.svg')

data=glass_s,testis_s
plt.figure(dpi=300)
ax=sns.violinplot(data=data, cut=0, palette=(oranges[0],blues[0]))

print('glass_s '+str(np.mean(glass_s))+"_+-_"+str(np.std(glass_s)))
print('testis_s '+str(np.mean(testis_s))+"_+-_"+str(np.std(testis_s)))

sns.despine()
#plt.ylim(0.0,100)
plt.title('Spontaneous activity')
plt.ylabel(label)

ax.set_xticklabels(['Glass', 'Testis'])

plt.savefig(str(title)+'_spon_glass_testis.jpg')
plt.savefig(str(title)+'_spon_glass_testis.svg')

data=glass_s,atp_1,atp_3,atp_10
plt.figure(dpi=300)
ax=sns.violinplot(data=data, cut=0, palette='Oranges_r')

print('glass_s '+str(np.mean(glass_s))+"_+-_"+str(np.std(glass_s)))
print('ATP_1 '+str(np.mean(atp_1))+"_+-_"+str(np.std(atp_1)))
print('ATP_3 '+str(np.mean(atp_3))+"_+-_"+str(np.std(atp_3)))
print('ATP_10 '+str(np.mean(atp_10))+"_+-_"+str(np.std(atp_10)))

sns.despine()
#plt.ylim(0.0,100)
plt.title('Spontaneous activity vs application')
plt.ylabel(label)

ax.set_xticklabels(['Glass', u'ATP 1 \u03bcMol',u'ATP 3 \u03bcMol','ATP 10 \u03bcMol'])

plt.savefig(str(title)+'_spon_glass_overview.jpg')
plt.savefig(str(title)+'_spon_glass_overview.svg')

data=testis_s,atp_1_testis,atp_3_testis,atp_10_testis
plt.figure(dpi=300)
ax=sns.violinplot(data=data, cut=0, palette=(blues))

print('testis_s '+str(np.mean(testis_s))+"_+-_"+str(np.std(testis_s)))
print('ATP_1_testis '+str(np.mean(atp_1_testis))+"_+-_"+str(np.std(atp_1_testis)))
print('ATP_3_testis '+str(np.mean(atp_3_testis))+"_+-_"+str(np.std(atp_3_testis)))
print('ATP_10_testis '+str(np.mean(atp_10_testis))+"_+-_"+str(np.std(atp_10_testis)))

print('n_testis_s '+str(len(testis_s)))
print('n_glass_s '+str(len(glass_s)))
print('n_ATP_1 '+str(len(atp_1)))
print('n_ATP_3 '+str(len(atp_3)))
print('n_ATP_10 '+str(len(atp_10)))
print('n_ATP_1_testis '+str(len(atp_1_testis)))
print('n_ATP_3_testis '+str(len(atp_3_testis)))
print('n_ATP_10_testis '+str(len(atp_10_testis)))


sns.despine()
#plt.ylim(0.0,100)
plt.title('Spontaneous activity vs application')
plt.ylabel(label)

ax.set_xticklabels(['Testis', u'ATP 1 \u03bcMol',u'ATP 3 \u03bcMol','ATP 10 \u03bcMol'])

plt.savefig(str(title)+'_spon_testis_overview.jpg')
plt.savefig(str(title)+'_spon_testis_overview.svg')

plt.figure(dpi=300)
plt.title('Glass vs Testis spontaneous Activity')
hist,edges=np.histogram(testis_s,bins=range(0,10))
plt.plot(edges[:-1],np.cumsum(hist*100/np.sum(hist)), c=blues[0], label='Testis')
hist,edges=np.histogram(glass_s,bins=range(0,10))
plt.plot(edges[:-1],np.cumsum(hist*100/np.sum(hist)), c=oranges[0], label='Glass')
plt.ylabel('Cumulative Number[%]')
plt.xlabel(label)
sns.despine()                                                                                                                                                    
plt.legend()

plt.savefig(str(title)+'_glass_testis_cum_hist.jpg')
plt.savefig(str(title)+'_glass_testis_cum_hist.svg')

plt.figure(dpi=300)
plt.title('Glass vs ATP')
hist,edges=np.histogram(glass_s,bins=range(0,10))
plt.plot(edges[:-1],np.cumsum(hist*100/np.sum(hist)), c=oranges[0], label='Glass')
hist,edges=np.histogram(atp_1,bins=range(0,10))
plt.plot(edges[:-1],np.cumsum(hist*100/np.sum(hist)),c=oranges[1], label=u'ATP 1 \u03bcMol')
hist,edges=np.histogram(atp_3,bins=range(0,10))
plt.plot(edges[:-1],np.cumsum(hist*100/np.sum(hist)),c=oranges[2], label=u'ATP 3 \u03bcMol')
hist,edges=np.histogram(atp_10,bins=range(0,10))
plt.plot(edges[:-1],np.cumsum(hist*100/np.sum(hist)),c=oranges[3], label=u'ATP 10 \u03bcMol')
plt.ylabel('Cumulative Number[%]')
plt.xlabel(label)
sns.despine()                                                                                                                                                    
plt.legend()

plt.savefig(str(title)+'_glass_atp_cum_hist.jpg')
plt.savefig(str(title)+'_glass_atp_cum_hist.svg')

plt.figure(dpi=300)
plt.title('Testis vs ATP')
hist,edges=np.histogram(glass_s,bins=range(0,10))
plt.plot(edges[:-1],np.cumsum(hist*100/np.sum(hist)), c=blues[0], label='Testis')
hist,edges=np.histogram(atp_1_testis,bins=range(0,10))
plt.plot(edges[:-1],np.cumsum(hist*100/np.sum(hist)),c=blues[1], label=u'ATP 1 \u03bcMol')
hist,edges=np.histogram(atp_3_testis,bins=range(0,10))
plt.plot(edges[:-1],np.cumsum(hist*100/np.sum(hist)),c=blues[2], label=u'ATP 3 \u03bcMol')
hist,edges=np.histogram(atp_10_testis,bins=range(0,10))
plt.plot(edges[:-1],np.cumsum(hist*100/np.sum(hist)),c=blues[3], label=u'ATP 10 \u03bcMol')
plt.ylabel('Cumulative Number[%]')
plt.xlabel(label)
sns.despine()                                                                                                                                                    
plt.legend()

plt.savefig(str(title)+'_testis_atp_cum_hist.jpg')
plt.savefig(str(title)+'_testis_atp_cum_hist.svg')

plt.figure(dpi=300)
plt.title('Glass vs Testis spontaneous Activity')
hist,edges=np.histogram(testis_s,bins=range(0,10))
plt.plot(edges[:-1],(hist*100/np.sum(hist)),c=blues[0], label='Testis')
hist,edges=np.histogram(glass_s,bins=range(0,10))
plt.plot(edges[:-1],(hist*100/np.sum(hist)),c=oranges[0], label='Glass')
plt.ylabel('Number[%]')
plt.xlabel(label)
sns.despine()                                                                                                                                                    
plt.legend()

plt.savefig(str(title)+'_glass_testis_hist.jpg')
plt.savefig(str(title)+'_glass_testis_hist.svg')

plt.figure(dpi=300)
plt.title('Glass vs ATP')
hist,edges=np.histogram(glass_s,bins=range(0,10))
plt.plot(edges[:-1],(hist*100/np.sum(hist)),c=oranges[0], label='Glass')
hist,edges=np.histogram(atp_1,bins=range(0,10))
plt.plot(edges[:-1],(hist*100/np.sum(hist)),c=oranges[1], label=u'ATP 1 \u03bcMol')
hist,edges=np.histogram(atp_3,bins=range(0,10))
plt.plot(edges[:-1],(hist*100/np.sum(hist)),c=oranges[2], label=u'ATP 3 \u03bcMol')
hist,edges=np.histogram(atp_10,bins=range(0,10))
plt.plot(edges[:-1],(hist*100/np.sum(hist)),c=oranges[3], label=u'ATP 10 \u03bcMol')
plt.ylabel('Number[%]')
plt.xlabel(label)
sns.despine()                                                                                                                                                    
plt.legend()

plt.savefig(str(title)+'_glass_atp_hist.jpg')
plt.savefig(str(title)+'_glass_atp_hist.svg')

plt.figure(dpi=300)
plt.title('Testis vs ATP')
hist,edges=np.histogram(glass_s,bins=range(0,10))
plt.plot(edges[:-1],(hist*100/np.sum(hist)),c=blues[0], label='Testis')
hist,edges=np.histogram(atp_1_testis,bins=range(0,10))
plt.plot(edges[:-1],(hist*100/np.sum(hist)),c=blues[1], label=u'ATP 1 \u03bcMol')
hist,edges=np.histogram(atp_3_testis,bins=range(0,10))
plt.plot(edges[:-1],(hist*100/np.sum(hist)),c=blues[2], label=u'ATP 3 \u03bcMol')
hist,edges=np.histogram(atp_10_testis,bins=range(0,10))
plt.plot(edges[:-1],(hist*100/np.sum(hist)),c=blues[3], label=u'ATP 10 \u03bcMol')
plt.ylabel('Number[%]')
plt.xlabel(label)
sns.despine()                                                                                                                                                    
plt.legend()

plt.savefig(str(title)+'_testis_atp_hist.jpg')
plt.savefig(str(title)+'_testis_atp_hist.svg')

print('Glass vs Testis,  '+str(stats.mannwhitneyu(glass_s, testis_s)[1]))
print('Glass vs ATP 1,  '+str(stats.mannwhitneyu(glass_s, atp_1)[1]*4))
print('Glass vs ATP 3,  '+str(stats.mannwhitneyu(glass_s, atp_3)[1]*4))
print('Glass vs ATP 10,  '+str(stats.mannwhitneyu(glass_s, atp_10)[1]*4))
print('ATP_1_testis vs Testis,  '+str(stats.mannwhitneyu(atp_1_testis, testis_s)[1]*4))
print('ATP_3_testis vs Testis,  '+str(stats.mannwhitneyu(atp_3_testis, testis_s)[1]*4))
print('ATP_10_testis vs Testis,  '+str(stats.mannwhitneyu(atp_10_testis, testis_s)[1]*4))
print('ATP_1_testis vs ATP_3_testis,  '+str(stats.mannwhitneyu(atp_1_testis, atp_3_testis)[1]*4))
print('ATP_1_testis vs ATP_10_testis,  '+str(stats.mannwhitneyu(atp_1_testis, atp_10_testis)[1]*4))
print('ATP_10_testis vs ATP_3_testis,  '+str(stats.mannwhitneyu(atp_10_testis, atp_3_testis)[1]*4))
print('ATP_1 vs ATP_3,  '+str(stats.mannwhitneyu(atp_1, atp_3)[1]*4))
print('ATP_1 vs ATP_10,  '+str(stats.mannwhitneyu(atp_1, atp_10)[1]*4))
print('ATP_10 vs ATP_3,  '+str(stats.mannwhitneyu(atp_10, atp_3)[1]*4))

