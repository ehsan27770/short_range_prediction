import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from benford import benford
# %% first data

df = pd.read_csv('./lightning_days/Säntis 2014 -2016 circle 30 km.dat',sep=';',index_col='date',parse_dates=['date'],infer_datetime_format=True)
df
df['amplitude'].plot.hist(bins=1000)
# %%

#cols = ['nano','st_x','st_y']
#df.drop(columns=cols)

# %%
df['amp_abs'] = np.abs(df['amplitude'])

find_pos = lambda x : np.abs(x) if x>0 else np.nan
find_neg = lambda x : np.abs(x) if x<0 else np.nan
df['amp_pos'] = df['amplitude'].apply(find_pos)
df['amp_neg'] = df['amplitude'].apply(find_neg)
df
# %%
#df.drop(['nano','st_x','st_y','nbloc','numloc','ki2','icloud','maxis','axisratio',])
to_drop = list(set(df.columns)-set(['amp_abs','amp_pos','amp_neg']))
df = df.drop(to_drop,axis=1)

# %% all lightnings types in day
df

df.info()

# %%
benford(list(df['amp_abs'].groupby([df.index.year,df.index.dayofyear]).count()),'all lightnings types in day')
# %%
benford(list(df['amp_pos'].groupby([df.index.year,df.index.dayofyear]).count()),'positive lightnings types in day')
# %%
benford(list(df['amp_neg'].groupby([df.index.year,df.index.dayofyear]).count()),'negative lightnings types in day')
# %%
for i in [2,5,10]:
    tmp = df[df['amp_abs']>i]['amp_abs']
    benford(list(tmp.groupby([tmp.index.year,tmp.index.dayofyear]).count()),'all lightning types above {} KA'.format(i))
# %%
for i in [2,5,10]:
    tmp = df[df['amp_abs']<i]['amp_abs']
    benford(list(tmp.groupby([tmp.index.year,tmp.index.dayofyear]).count()),'all lightning types below {} KA'.format(i))























# %% second data

df = pd.read_csv('./lightning_days/Saentis_Strokes_2000_2013.txt',sep=' ',header=None)
df
df['date'] = df[1] + ' ' + df[2]
df['date'] = pd.to_datetime(df['date'],infer_datetime_format=True)
df = df.set_index(['date'])
df = df.drop(columns=[1,2])
df = df.rename(columns={3:'nano',6:'st_y',7:'st_x',})
cols = [0,'nano','st_x','st_y',11,4,9,10,12,5,13,14,15,16,17,18,19,20,21,22]
df = df.drop(columns=cols)
df = df.rename(columns={8:'amplitude'})

df
df['amplitude'].plot.hist(bins=1000)


df['amp_abs'] = np.abs(df['amplitude'])

find_pos = lambda x : np.abs(x) if x>0 else np.nan
find_neg = lambda x : np.abs(x) if x<0 else np.nan
df['amp_pos'] = df['amplitude'].apply(find_pos)
df['amp_neg'] = df['amplitude'].apply(find_neg)
df = df.drop(columns='amplitude')
df

#%%
benford(list(df['amp_abs']),'')
# %%
benford(list(df['amp_pos']),'')
# %%
benford(list(df['amp_neg']),'')
# %%
benford(list(df['amp_abs'].groupby([df.index.year,df.index.dayofyear]).count()),'all lightnings types in day')
# %%
benford(list(df['amp_pos'].groupby([df.index.year,df.index.dayofyear]).count()),'positive lightnings types in day')
# %%
benford(list(df['amp_neg'].groupby([df.index.year,df.index.dayofyear]).count()),'negative lightnings types in day')
# %%
for i in [2,5,10]:
    tmp = df[df['amp_abs']>i]['amp_abs']
    benford(list(tmp.groupby([tmp.index.year,tmp.index.dayofyear]).count()),'all lightning types above {} KA'.format(i))
# %%
for i in [2,5,10]:
    tmp = df[df['amp_abs']<i*10]['amp_abs']
    benford(list(tmp.groupby([tmp.index.year,tmp.index.dayofyear]).count()),'all lightning types below {} KA'.format(i))
