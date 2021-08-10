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
# Effect of lightning type
for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
    #ax = benford(list(df['amp_abs'].groupby([df.index.year,df.index.dayofyear]).count()),f'all lightnings types in day {i}')
    f = benford(list(df['amp_abs'].groupby(df.index.floor('d')).count()),f'all lightnings types in day {i}')
    f.savefig(f'all_{i}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)
# %%
for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
    f = benford(list(df['amp_pos'].groupby(df.index.floor('d')).count()),f'positive lightnings types in day {i}')
    f.savefig(f'pos_{i}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)
# %%
for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
    f = benford(list(df['amp_neg'].groupby(df.index.floor('d')).count()),f'negative lightnings types in day {i}')
    f.savefig(f'neg_{i}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)




# %%
# Effect of counting period
for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
    f = benford(list(df['amp_abs'].groupby(pd.Grouper(freq='5D')).count()),f'all cloud-to-ground in day {i}')
# %%
for freq in ['1H','12H','1D','3D','7D','1M']:
    df = df1
    f = benford(list(df['amp_abs'].groupby(pd.Grouper(freq=freq)).count()),f'all lightning in {freq}')
    f.savefig(f'{freq}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)




# %%
# Effect of lightning destination
for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
    df = df[df['icloud']=='f']
    f = benford(list(df['amp_abs'].groupby(df.index.floor('d')).count()),f'all cloud-to-ground in day {i}')
    f.savefig(f'all_{i}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)
# %%
for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
    df = df[df['icloud']=='f']
    f = benford(list(df['amp_pos'].groupby(df.index.floor('d')).count()),f'positive cloud-to-ground in day {i}')
    f.savefig(f'pos_{i}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)
# %%
for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
    df = df[df['icloud']=='f']
    f = benford(list(df['amp_neg'].groupby(df.index.floor('d')).count()),f'negative cloud-to-ground in day {i}')
    f.savefig(f'neg_{i}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)
# %%
for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
    df = df[df['icloud']=='t']
    f = benford(list(df['amp_abs'].groupby(df.index.floor('d')).count()),f'all cloud-to-ground in day {i}')
    f.savefig(f'all_{i}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)
# %%
for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
    df = df[df['icloud']=='t']
    f = benford(list(df['amp_pos'].groupby(df.index.floor('d')).count()),f'positive cloud-to-ground in day {i}')
    f.savefig(f'pos_{i}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)
# %%
for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
    df = df[df['icloud']=='t']
    f = benford(list(df['amp_neg'].groupby(df.index.floor('d')).count()),f'negative cloud-to-ground in day {i}')
    f.savefig(f'neg_{i}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)


# %%
# Effect of multiplication
for i,df in enumerate([df1,df2,df3]):
    list_of_data = list(df['amp_abs'].groupby(df.index.floor('d')).count())
    perturbed = []
    rhos = np.arange(0.9,12,0.01)
    for rho in rhos:
        jsd, kld = normalize(list_of_data,rho)
        perturbed.append(jsd)

    plt.figure(figsize=(15,10))
    plt.plot(rhos, perturbed)
# %%
for i,df in enumerate([df1,df2,df3]):
    list_of_data = list(df['amp_pos'].groupby(df.index.floor('d')).count())
    perturbed = []
    rhos = np.arange(0.9,12,0.01)
    for rho in rhos:
        jsd, kld = normalize(list_of_data,rho)
        perturbed.append(jsd)

    plt.figure(figsize=(15,10))
    plt.plot(rhos, perturbed)

# %%
for i,df in enumerate([df1,df2,df3]):
    list_of_data = list(df['amp_neg'].groupby(df.index.floor('d')).count())
    perturbed = []
    rhos = np.arange(0.9,12,0.01)
    for rho in rhos:
        jsd, kld = normalize(list_of_data,rho)
        perturbed.append(jsd)

    plt.figure(figsize=(15,10))
    plt.plot(rhos, perturbed)

# %%
for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
    f = benford(df['amp_abs'],f'absolute amplitude {i}')
    f.savefig(f'all_{i}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)
# %%
for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
    f = benford(df['amp_pos'],f'positive amplitude {i}')
    f.savefig(f'pos_{i}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)
# %%
for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
    f = benford(df['amp_neg'],f'negative amplitude {i}')
    f.savefig(f'neg_{i}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)
