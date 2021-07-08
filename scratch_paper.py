import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from benford import benford, normalize
# %%
find_pos = lambda x : np.abs(x) if x>0 else np.nan
find_neg = lambda x : np.abs(x) if x<0 else np.nan
# %%
dfs = [pd.read_csv(f'data/data_2000-2004_Impact_System/strokedata_200{i}.asc',sep='|',names=['num','date','nano','st_y','st_x','amplitude','icloud','flash'],index_col='date',parse_dates=['date']) for i in range(5)]
df1 = pd.concat(dfs,join='inner',ignore_index=False)

df1['amp_abs'] = np.abs(df1['amplitude'])
df1['amp_pos'] = df1['amplitude'].apply(find_pos)
df1['amp_neg'] = df1['amplitude'].apply(find_neg)
df1
# %%
dfs = [pd.read_csv(f'data/data_2010-2014_LS7000_System/strokedata_201{i}.asc',sep='|',names=['num','date','nano','st_y','st_x','amplitude','icloud','flash'],index_col='date',parse_dates=['date']) for i in range(5)]
df2 = pd.concat(dfs,join='inner',ignore_index=False)

df2['amp_abs'] = np.abs(df2['amplitude'])
df2['amp_pos'] = df2['amplitude'].apply(find_pos)
df2['amp_neg'] = df2['amplitude'].apply(find_neg)
df2
# %%
dfs = [pd.read_csv(f'data/data_2016-2020_LS7002_System/strokedata_20{i}.asc',sep='|',names=['num','date','nano','st_y','st_x','amplitude','icloud','flash'],index_col='date',parse_dates=['date']) for i in range(16,21)]
df3 = pd.concat(dfs,join='inner',ignore_index=False)

df3['amp_abs'] = np.abs(df3['amplitude'])
df3['amp_pos'] = df3['amplitude'].apply(find_pos)
df3['amp_neg'] = df3['amplitude'].apply(find_neg)
df3

# %%
for i,df in enumerate([df1,df2,df3]):
    ax = benford(list(df['amp_abs'].groupby([df.index.year,df.index.dayofyear]).count()),'all lightnings types in day')
# %%
for i,df in enumerate([df1,df2,df3]):
    list_of_data = list(df['amp_abs'].groupby([df.index.year,df.index.dayofyear]).count())
    perturbed = []
    rhos = np.arange(0.9,12,0.01)
    for rho in rhos:
        jsd, kld = normalize(list_of_data,rho)
        perturbed.append(jsd)

    plt.figure(figsize=(15,10))
    plt.plot(rhos, perturbed)
# %%
for i,df in enumerate([df1,df2,df3]):
    benford(list(df['amp_pos'].groupby([df.index.year,df.index.dayofyear]).count()),'positive lightnings types in day')
# %%
for i,df in enumerate([df1,df2,df3]):
    list_of_data = list(df['amp_pos'].groupby([df.index.year,df.index.dayofyear]).count())
    perturbed = []
    rhos = np.arange(0.9,12,0.01)
    for rho in rhos:
        jsd, kld = normalize(list_of_data,rho)
        perturbed.append(jsd)

    plt.figure(figsize=(15,10))
    plt.plot(rhos, perturbed)
# %%
for i,df in enumerate([df1,df2,df3]):
    benford(list(df['amp_neg'].groupby([df.index.year,df.index.dayofyear]).count()),'negative lightnings types in day')
# %%
for i,df in enumerate([df1,df2,df3]):
    list_of_data = list(df['amp_neg'].groupby([df.index.year,df.index.dayofyear]).count())
    perturbed = []
    rhos = np.arange(0.9,12,0.01)
    for rho in rhos:
        jsd, kld = normalize(list_of_data,rho)
        perturbed.append(jsd)

    plt.figure(figsize=(15,10))
    plt.plot(rhos, perturbed)
# %%
for i,df in enumerate([df1,df2,df3]):
    df = df[df['icloud']=='f']
    ax = benford(list(df['amp_pos'].groupby([df.index.year,df.index.dayofyear]).count()),'all lightnings types in day')


# %%
for i,df in enumerate([df1,df2,df3]):
    df = df[df['icloud']=='f']
    ax = benford(list(df['amp_neg'].groupby([df.index.year,df.index.dayofyear]).count()),'all lightnings types in day')
# %%
a = df1['amp_abs'].groupby([df1.index.year,df1.index.dayofyear]).count()
a.median()
a.mean()
a.hist(bins=100)
a
# %%
for i,df in enumerate([df1,df2,df3]):
    ax = benford(df['amp_abs'],'absolute amplitude')
# %%
for i,df in enumerate([df1,df2,df3]):
    ax = benford(df['amp_pos'],'positive amplitude')
# %%
for i,df in enumerate([df1,df2,df3]):
    ax = benford(df['amp_neg'],'negative amplitude')
# %%
for i,df in enumerate([df1,df2,df3]):
    ax = benford(df['nano'],'duration')
