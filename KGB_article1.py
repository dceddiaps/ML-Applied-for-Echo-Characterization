# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 09:25:54 2022

@author: userDCPS
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings(action= 'ignore')

data = pd.read_csv(r"C:\DCPS\Mestrado_DOT\ML KGB\kgb_data_ecos.txt")
data.rename(columns = {'SLOPE_DIR':'ASPECT'}, inplace = True)
data = data[data.SLOPE < 50]

# Eliminando dados duplicados
print('Amount of rows in original dataset:',len(data))
duplicateDFRow = data[data.duplicated(subset=['Z','BS','SLOPE','ECO'])]
print('Amount of duplicated rows:', len(duplicateDFRow))
print('% filtered:', np.round(len(duplicateDFRow)/len(data)*100,3))
data.drop(axis=0, index=duplicateDFRow.index, inplace=True)
data = data.reset_index(drop=True)
print('Lenght of new dataset:',len(data),'\n')
del duplicateDFRow

print('% of Echo 1:',np.round((len(np.where(data.ECO==1)[0])/len(data))*100,3))
print('% of Echo 2:',np.round((len(np.where(data.ECO==2)[0])/len(data))*100,3))
print('% of Echo 3:',np.round((len(np.where(data.ECO==3)[0])/len(data))*100,3),'\n')

# Gaussian Fit
def GF(data):
    mean = data.mean()
    std  = data.std()
    x = np.linspace(mean - 3*std, mean + 3*std, 200)
    y = stats.norm.pdf(x, mean, std)
    return np.round(mean,3), np.round(std,3), x, y

mean1,std1,x1,y1 = GF(data[data.ECO==1].BS)
mean2,std2,x2,y2 = GF(data[data.ECO==2].BS)
mean3,std3,x3,y3 = GF(data[data.ECO==3].BS)


# Plot

plt.figure(figsize=(10,5))
kde1 = sns.kdeplot(data[data.ECO==1].BS, color='black', Label='Echo 1',linewidth=1.5)
kde2 = sns.kdeplot(data[data.ECO==2].BS, color='blue', Label='Echo 2',linewidth=1.5)
kde3 = sns.kdeplot(data[data.ECO==3].BS, color='yellow', Label='Echo 3',linewidth=1.5)

plt.plot(x1,y1, color='black', linestyle='dashed', 
         label=f'Echo 1 Gaussian N({mean1},{std1})')
plt.plot(x2,y2, color='blue', linestyle='dashed',
         label=f'Echo 2 Gaussian N({mean2},{std2})')
plt.plot(x3,y3, color='yellow', linestyle='dashed',
         label=f'Echo 3 Gaussian N({mean3},{std3})')
plt.xlim(-25,-12)
plt.title('Backscatter intensity distribution for each echo-type',fontweight='bold')
plt.xlabel('Backscatter Intensity (dB)')
plt.ylabel('Kernel Density Estimator')
plt.grid()
plt.legend()
plt.show()


##









