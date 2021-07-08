import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from benford import benford
# %% first data

df1 = pd.read_csv('./data/Säntis 2014 -2016 circle 30 km.dat',sep=';',index_col='date',parse_dates=['date'],infer_datetime_format=True)
df1
df1['amplitude'].plot.hist(bins=1000)
# %%

#cols = ['nano','st_x','st_y']
#df.drop(columns=cols)

# %%
df1['amp_abs'] = np.abs(df1['amplitude'])

find_pos = lambda x : np.abs(x) if x>0 else np.nan
find_neg = lambda x : np.abs(x) if x<0 else np.nan
df1['amp_pos'] = df1['amplitude'].apply(find_pos)
df1['amp_neg'] = df1['amplitude'].apply(find_neg)
df1
# %%
for col in df1.columns:
    try:
        benford(df1[col],f'{col}')
    except:
        continue
# %%
#df.drop(['nano','st_x','st_y','nbloc','numloc','ki2','icloud','maxis','axisratio',])
to_drop = list(set(df1.columns)-set(['amp_abs','amp_pos','amp_neg']))
df1 = df1.drop(to_drop,axis=1)

# %% all lightnings types in day
df1

df1.info()

# %%
benford(list(df1['amp_abs'].groupby([df1.index.year,df1.index.dayofyear]).count()),'all lightnings types in day')
# %%
benford(list(df1['amp_pos'].groupby([df1.index.year,df1.index.dayofyear]).count()),'positive lightnings types in day')
# %%
benford(list(df1['amp_neg'].groupby([df1.index.year,df1.index.dayofyear]).count()),'negative lightnings types in day')
# %%
for i in [2,5,10]:
    tmp = df1[df1['amp_abs']>i]['amp_abs']
    benford(list(tmp.groupby([tmp.index.year,tmp.index.dayofyear]).count()),'all lightning types above {} KA'.format(i))
# %%
for i in [2,5,10]:
    tmp = df1[df1['amp_abs']<i]['amp_abs']
    benford(list(tmp.groupby([tmp.index.year,tmp.index.dayofyear]).count()),'all lightning types below {} KA'.format(i))























# %% second data

df2 = pd.read_csv('./data/Saentis_Strokes_2000_2013.txt',sep=' ',header=None)
df2
df2['date'] = df2[1] + ' ' + df2[2]
df2['date'] = pd.to_datetime(df2['date'],infer_datetime_format=True)
df2 = df2.set_index(['date'])
df2 = df2.drop(columns=[1,2])
df2 = df2.rename(columns={3:'nano',6:'st_y',7:'st_x',})
cols = [0,'nano','st_x','st_y',11,4,9,10,12,5,13,14,15,16,17,18,19,20,21,22]
df2 = df2.drop(columns=cols)
df2 = df2.rename(columns={8:'amplitude'})

df2
df2['amplitude'].plot.hist(bins=1000)


df2['amp_abs'] = np.abs(df2['amplitude'])

find_pos = lambda x : np.abs(x) if x>0 else np.nan
find_neg = lambda x : np.abs(x) if x<0 else np.nan
df2['amp_pos'] = df2['amplitude'].apply(find_pos)
df2['amp_neg'] = df2['amplitude'].apply(find_neg)
df2 = df2.drop(columns='amplitude')
df2

# %%
for col in df2.columns:
    try:
        benford(df2[col],f'{col}')
    except:
        continue
#%%
benford(list(df2['amp_abs']),'')
# %%
benford(list(df2['amp_pos']),'')
# %%
benford(list(df2['amp_neg']),'')
# %%
benford(list(df2['amp_abs'].groupby([df2.index.year,df2.index.dayofyear]).count()),'all lightnings types in day')
# %%
benford(list(df2['amp_pos'].groupby([df2.index.year,df2.index.dayofyear]).count()),'positive lightnings types in day')
# %%
benford(list(df2['amp_neg'].groupby([df2.index.year,df2.index.dayofyear]).count()),'negative lightnings types in day')
# %%
for i in [2,5,10]:
    tmp = df2[df2['amp_abs']>i]['amp_abs']
    benford(list(tmp.groupby([tmp.index.year,tmp.index.dayofyear]).count()),'all lightning types above {} KA'.format(i))
# %%
for i in [2,5,10]:
    tmp = df2[df2['amp_abs']<i*10]['amp_abs']
    benford(list(tmp.groupby([tmp.index.year,tmp.index.dayofyear]).count()),'all lightning types below {} KA'.format(i))
