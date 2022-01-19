from pykdtree.kdtree import KDTree
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import time

# IMPORTANDO DATASET DE VALIDAÇÃO----------------------------------------------

data = pd.read_csv(r"C:\DCPS\Mestrado_DOT\ML ALM\dataset_alm_ecos.txt",
                   delim_whitespace=False)
data.rename(columns = {
    'x':'X',
    'y':'Y',
    'SLOPE_DIR':'ASPECT'
    }, inplace = True)

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

# IMPORTANDO DATASET DE EXTRAPOLAÇÃO-------------------------------------------

import pandas as pd
import numpy as np

data_all = pd.read_csv(r"C:\DCPS\Mestrado_DOT\ML ALM\dataset_alm_all.txt",
                   delim_whitespace=False)
data_all = data_all[data_all.SLOPE < 60]
data_all.rename(columns = {'SLOPE_DIR':'ASPECT'}, inplace = True)
data_all = data_all.reset_index(drop=True)
data_all = data_all[data_all.SLOPE < 60]

# Eliminando dados duplicados
print('Qtd de rows originais:',len(data_all))
duplicateDFRow = data_all[data_all.duplicated(subset=['Z','BS','SLOPE','ASPECT'])]
print('Qtd de rows duplicadas:', len(duplicateDFRow))
print('% do dataset filtrado:', np.round(len(duplicateDFRow)/len(data_all)*100,3))
data_all.drop(axis=0, index=duplicateDFRow.index, inplace=True)
data_all = data_all.reset_index(drop=True)
print('Qtd de rows do novo dataset:',len(data_all),'\n')
del duplicateDFRow

# REMOVENDO DADOS DE VALIDAÇÃO DENTRO DOS DADOS DE TESTE-----------------------

import pandas as pd

temp = pd.concat([data,data_all])
duplicateDFRow = temp[temp.duplicated(subset=['Z','BS','SLOPE','ASPECT'],keep='first')]
data = data.drop(duplicateDFRow.iloc[np.where(duplicateDFRow.index < len(data))[0]].index)
data_all = data_all.drop(duplicateDFRow.iloc[np.where(duplicateDFRow.index > len(data))[0]].index)
print('Qtd de rows duplicadas eliminadas entre os datasets de validação e teste:', len(duplicateDFRow),'\n')
data_all = data_all.reset_index(drop=True)
del temp; del duplicateDFRow

# #paths
# p_d1 = r"C:\DCPS\Mestrado_DOT\ML ALM\dataset_alm_all.txt"
# p_d2 = r"C:\DCPS\Mestrado_DOT\ML ALM\dataset_alm_ecos.txt"

# #load
# df1 = pd.read_csv(p_d1, usecols=['X','Y'], dtype='float32')
# df2 = pd.read_csv(p_d2, usecols=['x','y'], dtype='float32')

# #PYKDTREE
# kd_tree = KDTree(df1.values)
# start = time.time()
# dist, idx = kd_tree.query(df2.values, k=1)
# end = time.time()
# print(end - start)

df = data_all[['X','Y']]

#PYKDTREE
kd_tree = KDTree(data_all[['X','Y']].values)
start = time.time()
dist, idx = kd_tree.query(df.values, k=1)
end = time.time()
print(end - start)

# #SCIPY
# tree = cKDTree(data_all[['X','Y']].values)
# start_ = time.time()
# dist_, idx_ = tree.query(df.values, k=1)
# end_ = time.time()
# print(end_ - start_)

df['Z'] = data_all.Z.iloc[idx]
df['BS'] = data_all.BS.iloc[idx]
df['SLOPE'] = data_all.SLOPE.iloc[idx]
df['ASPECT'] = data_all.ASPECT.iloc[idx]
df = df.reset_index(drop=True)

# PLOTS DA DISTRIBUIÇÃO DAS FEATURES DA DISTANCIA EUCLIDIANA VS. KDTREE--------

import pandas as pd
from sklearn.preprocessing import StandardScaler

z1 = pd.DataFrame(StandardScaler().fit_transform(df.Z.values.reshape(-1, 1)),columns=['Z1n'])
bs1 = pd.DataFrame(StandardScaler().fit_transform(df.BS.values.reshape(-1, 1)),columns=['BS1n'])
sl1 = pd.DataFrame(StandardScaler().fit_transform(df.SLOPE.values.reshape(-1, 1)),columns=['SL1n'])
# do1 = pd.DataFrame(StandardScaler().fit_transform(df.distOUTFALL.values.reshape(-1, 1)),columns=['DO1n'])
# dg1 = pd.DataFrame(StandardScaler().fit_transform(df.distGLACIER.values.reshape(-1, 1)),columns=['DG1n'])
as1 = pd.DataFrame(StandardScaler().fit_transform(df.ASPECT.values.reshape(-1, 1)),columns=['AS1n'])
z2 = pd.DataFrame(StandardScaler().fit_transform(data_all.Z.values.reshape(-1, 1)),columns=['Z2n'])
bs2 = pd.DataFrame(StandardScaler().fit_transform(data_all.BS.values.reshape(-1, 1)),columns=['BS2n'])
sl2 = pd.DataFrame(StandardScaler().fit_transform(data_all.SLOPE.values.reshape(-1, 1)),columns=['SL2n'])
# do2 = pd.DataFrame(StandardScaler().fit_transform(data_all.distOUTFALL.values.reshape(-1, 1)),columns=['DO2n'])
# dg2 = pd.DataFrame(StandardScaler().fit_transform(data_all.distGLACIER.values.reshape(-1, 1)),columns=['DG2n'])
as2 = pd.DataFrame(StandardScaler().fit_transform(data_all.ASPECT.values.reshape(-1, 1)),columns=['AS2n'])

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

sns.kdeplot(z1.Z1n.sample(70000,random_state=0,replace=True), shade=False, 
            label=f'Z (m), std_dev = {df.Z.values.std().round(2)}, mean = {df.Z.values.mean().round(2)}')
sns.kdeplot(bs1.BS1n.sample(70000,random_state=0,replace=True), shade=False, 
            label=f'BS (dB), std_dev = {df.BS.values.std().round(2)}, mean = {df.BS.values.mean().round(2)}', bw=0.17)
sns.kdeplot(sl1.SL1n.sample(70000,random_state=0,replace=True), shade=False, 
            label=f'SLOPE (degree), std_dev = {df.SLOPE.values.std().round(2)}, mean = {df.SLOPE.values.mean().round(2)}')
sns.kdeplot(as1.AS1n.sample(70000,random_state=0,replace=True), shade=False, 
            label=f'ASPECT (degree), std_dev = {df.ASPECT.values.std().round(2)}, mean = {df.ASPECT.values.mean().round(2)}')

sns.kdeplot(z2.Z2n.sample(70000,random_state=0,replace=True), 
            shade=False, label=f'Z (m), std_dev = {data_all.Z.values.std().round(2)}, mean = {data_all.Z.values.mean().round(2)}',
            color = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
            linestyle = 'dashed')
sns.kdeplot(bs2.BS2n.sample(70000,random_state=0,replace=True), 
            shade=False, label=f'BS (dB), std_dev = {data_all.BS.values.std().round(2)}, mean = {data_all.BS.values.mean().round(2)}', bw=0.17,
            color = (1.0, 0.4980392156862745, 0.054901960784313725),
            linestyle = 'dashed')
sns.kdeplot(sl2.SL2n.sample(70000,random_state=0,replace=True), 
            shade=False, label=f'SLOPE (degree), std_dev = {data_all.SLOPE.values.std().round(2)}, mean = {data_all.SLOPE.values.mean().round(2)}',
            color = (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
            linestyle = 'dashed')
sns.kdeplot(as2.AS2n.sample(70000,random_state=0,replace=True), 
            shade=False, label=f'ASPECT (degree), std_dev = {data_all.ASPECT.values.std().round(2)}, mean = {data_all.ASPECT.values.mean().round(2)}',
            color = (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
            linestyle = 'dashed')

plt.xlim(-2.7,3.5)
plt.ylim=(0,0.9)
plt.title('Normalized distribution of features using K-D-Tree (normal line style) versus Euclidean Distance method (dashed)',
          fontweight='bold', fontsize=12)
plt.xlabel('Normalized values')
plt.ylabel('Kernel Density Estimation')
plt.legend(prop={'size': 9})
plt.grid()

sum(data_all.Z - df.Z)








