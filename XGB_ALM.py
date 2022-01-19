# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 09:16:21 2021

@author: userDCPS
"""

# IMPORTANDO DATASET1----------------------------------------------------------

import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\DCPS\Mestrado_DOT\ML ALM\dataset_alm_ecos.txt",
                   delim_whitespace=False)
data.rename(columns = {
    'x':'X',
    'y':'Y',
    'SLOPE_DIR':'ASPECT'
    }, inplace = True)
data.line = data.line.str.replace(' ','')
data = data.sort_values(by='line').reset_index(drop=True)

# Deixando ecos no padrão KGB
data.eco[np.where(data.eco==4)[0]] = 3;
data.eco[np.where(data.eco==1)[0]] = 33;
data.eco[np.where(data.eco==3)[0]] = 11;
data.eco[np.where(data.eco==33)[0]] = 3;
data.eco[np.where(data.eco==11)[0]] = 1;

data = data.drop(np.where(data.eco==0)[0])
data = data.reset_index(drop=True)
# dado já com slope filtrado   
data = data[data.SLOPE < 60]

# Eliminando dados duplicados
print('Qtd de rows originais:',len(data))
duplicateDFRow = data[data.duplicated(subset=['Z','BS','SLOPE','eco'])]
print('Qtd de rows duplicadas:', len(duplicateDFRow))
print('% do dataset filtrado:', np.round(len(duplicateDFRow)/len(data)*100,3))
data.drop(axis=0, index=duplicateDFRow.index, inplace=True)
data = data.reset_index(drop=True)
print('Qtd de rows do novo dataset:',len(data),'\n')

print('% of Echo 1:',np.round((len(np.where(data.eco==1)[0])/len(data))*100,3))
print('% of Echo 2:',np.round((len(np.where(data.eco==2)[0])/len(data))*100,3))
print('% of Echo 3:',np.round((len(np.where(data.eco==3)[0])/len(data))*100,3),'\n')
del duplicateDFRow


# XGB STRESS TEST (a) methodology----------------------------------------------

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

# Estabelecimento e separação das features e target.
features = ['BS','Z','SLOPE','distOUTFALL','distGLACIER','ASPECT']
target = ['eco']
X = data[features]
y = data[target]
    
accu_a=[]

for i in tqdm(range(1,1000,1)):
  
    train_size=i/1000
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0,
                                                            train_size=train_size, 
                                                            stratify=y.eco)

    # Criação da Pipeline de processamento: normalização e modelo XGB default.
    pipe = Pipeline(steps = [
        ('scalar', StandardScaler()),
        ('model', XGBClassifier(eval_metric='error'))
        ])    

    pipe.fit(X_train, y_train.values.ravel())
    y_pred = pipe.predict(X_valid)
    y_pred = pd.DataFrame(y_pred,columns=['ECO'])
    accu_a.append(metrics.balanced_accuracy_score(y_valid,y_pred)*100)

import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.plot(np.arange(1,1000,1)/10,accu_a,color='black')
plt.title('Accuracy versus % of dataset used in training',fontweight='bold',fontsize='12')
plt.xlabel('Dataset used in training (%)')
plt.ylabel('Accuracy (%)')
plt.grid()


# CALCULANDO COMBINAÇÕES DE 1 A 51 (b) methodology-----------------------------

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from itertools import combinations
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import time
from sklearn import metrics
from operator import itemgetter
import warnings
warnings.filterwarnings('ignore')

linhas = np.unique(data.line)
features = ['BS','Z','SLOPE','distOUTFALL','distGLACIER','ASPECT']
target = ['eco']

report = pd.DataFrame(index=linhas, columns = np.arange(1,len(linhas)))
cum = pd.DataFrame()
melhores_linhas = []

for j in range(len(linhas)-1):
    if j==0:
        res_linhas=linhas
    if j != 0:
        res_linhas = [ ele for ele in linhas ]
        for a in melhores_linhas:
          if a in linhas:
            res_linhas.remove(a)
        
    for i in tqdm(range(len(res_linhas))):
        
        X = pd.concat([cum,data.iloc[np.where(data.line==res_linhas[i])[0]]])[features]
        y = pd.concat([cum,data.iloc[np.where(data.line==res_linhas[i])[0]]])[target]
    
        pipe = Pipeline(steps = [
            ('scalar', StandardScaler()),
            ('model', XGBClassifier(eval_metric='error'))
            ], verbose=False)
        
        pipe.fit(X, y.values.ravel())
        y_pred = pipe.predict(data.drop(X.index)[features])
        report[j+1].iloc[report.index.get_loc(res_linhas[i])] = metrics.balanced_accuracy_score(data.drop(X.index)[target], y_pred)
    melhores_linhas.append(report.index[np.where(report[j+1] == report[j+1].sort_values(ascending=False)[0])[0][0]])
    print('\n',melhores_linhas)
    print(report.iloc[np.where(report[j+1] == report[j+1].sort_values(ascending=False)[0])[0][0]][j+1],'\n')
    print(f'{len(melhores_linhas)} linhas já foram escolhidas. Restam {len(res_linhas)}.')
    cum = pd.concat([cum,data.iloc[np.where(data.line==melhores_linhas[-1])[0]]])

# Plot
accu = []
size = []
for i in range(len(melhores_linhas)):
    accu.append(report.iloc[np.where(report.index==melhores_linhas[i])[0][0]].max()*100)
    size.append(len(np.where(data.line==melhores_linhas[i])[0]))    

import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(figsize=(8,4))
ax1.plot(np.arange(0,len(linhas)-1)+1,accu,color='black', label='Accuracy')
ax1.set_ylabel('Accuracy (%)')
ax1.set_xlabel('Number of seismic lines in dataset')
plt.grid()
ax2 = ax1.twinx()
ax2.plot(np.arange(0,len(linhas)-1)+1,((np.cumsum(size)/len(data))*100)[:len(linhas)-1],
         label='% of dataset used')
ax2.set_ylabel('% of dataset used') 
plt.title('Acurracy and % of dataset versus lines used',fontweight='bold',fontsize='12')

handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
fig.legend(handles, labels, loc=(0.62,0.12))

plt.show()

# Tabela final
tab = pd.concat([pd.DataFrame(np.round(accu,2)),
                 pd.DataFrame((np.round((np.cumsum(size)/len(data))*100,2)))],
                axis=1)[:32]
tab.columns=['accu','size']
# tab.to_csv(r"C:\DCPS\Mestrado_DOT\ML ALM\tab_artigo1.txt",
#                     index=True,header=True,sep=',')

# EXPORTAÇÃO DAS LINHAS

melhores_linhas = ['OP32-Almirantado','OP32-Almirantado16_003','OP32-Almirantado16_002',
                   'OP32-Almirantado13_000','Alm-L2b.sgy','OP31-L5.004','OP32-Almirantado4',
                   'ALMIRANTADO-deception_001','OP32-Almirantado16_001','OP32-Almirantado14',
                   'OP32-Reigeorge10','ALM-P5','Perfil1P-OP34','Perfil1K-OP34','OP32-Almirantado16_004',
                   'OP32-Almirantado16_000_001','OP32-Reigeorge_001','OP32-Almirantado6','OP32-Almirantado5',
                   'Perfil5D-OP34','AlmL2-1.001','OP32-Almirantado16_005']
root_output = r"C:\DCPS\Mestrado_DOT\ML ALM\linhas_metodologia_b\\"
for i in melhores_linhas[:22]:
    for j in np.unique(data.eco):
        globals()[f'%s_eco{int(j)}' % i] = data.iloc[np.where((data.line==i) & 
                                                         (data.eco==int(j)))[0]]
        globals()[f'%s_eco{int(j)}' % i][['X','Y']].to_csv(root_output+i+f'_eco{int(j)}'+'.txt',
                                                header=False, index=False)

pd.read_csv(root_output+i+f'_eco{int(j)}'+'.txt')



