#%%
import numpy as np
import pandas as pd
from benford import benford
import matplotlib.pyplot as plt
%load_ext autoreload
%autoreload 2

#%%


dfs = [pd.read_csv(r"./lightning_data/{}_data.txt".format(i),sep=';',low_memory=False) for i in range(2007,2021)]


data = pd.concat(dfs,join='outer',ignore_index=True,verify_integrity=True)


data = data.drop_duplicates()

#clean_data.to_csv("./lightning_data/data.csv",index=False)


# %%
for col in list(data.columns)[2:]:
    data[col] = pd.to_numeric(data[col],errors='coerce',downcast='unsigned')
data = data.dropna()
data = data.reset_index(drop=True)

# %%
data

data['time'] = pd.to_datetime(data.time.astype('str'),format='%Y%m%d%H%M')

data['bre0'] = data['brefarz0'] + data['brecloz0']
data['onoff'] = np.sign(data['bre0'])

data = data.set_index(['time'])

data.to_csv('./lightning_data/data.csv')







# %% start

data = pd.read_csv('./lightning_data/data.csv',sep=',',index_col='time',parse_dates=['time'],infer_datetime_format=True)
data = data.drop(data[data.index.year==2021].index)
data
#set(list(map(type,data['time'])))
# %%
#data_by_year = data.groupby([data.index.year,'stn']).agg({'bre0':{'light_count':np.sum},'onoff':{'light_day':np.sum}})
data_by_year = data.groupby([data.index.year,'stn']).agg(light_count=('bre0',np.sum))
data_by_year.loc[2007]
# %%
data_by_month = data.groupby([data.index.month,'stn']).agg(light_count=('bre0',np.sum))
data_by_month
#,light_day=('onoff',np.sum)
# %%
data_by_day = data.groupby([data.index.dayofyear,'stn']).agg(light_count=('bre0',np.sum))
data_by_day
# %%
data_by_year_month = data.groupby([data.index.year,data.index.month,'stn']).agg(light_count=('bre0',np.sum))
data_by_year_month
# %%
data_by_year_day = data.groupby([data.index.year,data.index.dayofyear,'stn']).agg(light_count=('bre0',np.sum))
data_by_year_day
# %%
#data.groupby(data.index.year).agg({'bre0':np.sum}).expanding().agg({'bre0':np.sum})
data_by_year_day.index
# %%
stations = list(data['stn'].unique())

for s in stations:
    benford(data.loc[data.stn==s,'bre0'],s)

# %%
df = data.loc[data.stn=='SAE',['bre0']]
# %%
tmp = df.groupby(df.index.year).agg(light_count=('bre0','sum'))
plt.plot(tmp.index,tmp['light_count'])
plt.title('lightning over the years')
# %%
tmp = df.groupby(df.index.month).agg(light_count=('bre0','sum'))
plt.plot(tmp.index,tmp['light_count'])
plt.title('lightning over the month')
# %%
tmp = df.groupby([df.index.year,df.index.month]).agg(light_count=('bre0','sum'))
for i in range(2007,2021):
    plt.plot(tmp.loc[i].index,tmp.loc[i]['light_count'],'-*')
plt.legend(list(range(2007,2021)))
plt.title('lightning over the month')
# %%
tmp = df.groupby([df.index.year,df.index.dayofyear]).agg(light_count=('bre0','sum'))
tmp['light_day'] = list(map(np.sign,tmp['light_count']))
tmp = tmp.groupby(tmp.index.get_level_values(0)).sum()
tmp
plt.plot(tmp.index,tmp['light_day'])
plt.title('lightning days over the year')



# %%
tmp = df.groupby([df.index.year,df.index.dayofyear]).agg(light_count=('bre0','sum'))
tmp['light_day'] = list(map(np.sign,tmp['light_count']))
tmp = tmp.groupby(tmp.index.get_level_values(1)).sum()
tmp
plt.plot(tmp.index,tmp['light_day'])
plt.title('lightning days over through years')

# %%
tmp = df.groupby([df.index.year,df.index.dayofyear]).agg(light_count=('bre0','sum'))
benford(tmp.unstack([0,-1]),'lightning per day')

# %%
tmp = df.groupby([df.index.year,df.index.month]).agg(light_count=('bre0','sum'))
benford(tmp.unstack([0,-1]),'lightning per month')
# %%
tmp = df.groupby([df.index.year]).agg(light_count=('bre0','sum'))
benford(tmp.unstack([-1]),'lightning per year')
