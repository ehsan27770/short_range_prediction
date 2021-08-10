import numpy as np
import pandas as pd
from datetime import datetime,timedelta
import random
from benford import benford

# %%
a = 60
start = datetime(year=2021,month=1,day=1,hour=0,minute=0,second=0,microsecond=0)
end = datetime(year = 2026,month=1,day=1,hour=0,minute=0,second=0,microsecond=0)
# %%
#np.random.binomial()
#np.random.poisson()
#np.random.rand()
# %%
mean = .5 #min
# %% rayleigh https://en.wikipedia.org/wiki/Rayleigh_distribution
scale = mean/1.253
events = []
temp = start
for i in range(1000000):
    if temp < end:
        events.append(temp)
    else:
        break
    temp = temp + timedelta(minutes=np.random.rayleigh(scale=scale))

col = [1 for i in range(len(events))]
df = pd.DataFrame(col,index=events)
f = benford(list(df.groupby(pd.Grouper(freq='1D')).count()[0]),f'all lightning in 1D')

# %% exponential https://en.wikipedia.org/wiki/Exponential_distribution
scale = mean
events = []
temp = start
for i in range(1000000):
    if temp < end:
        events.append(temp)
    else:
        break
    temp = temp + timedelta(minutes=np.random.exponential(scale=scale))

col = [1 for i in range(len(events))]
df = pd.DataFrame(col,index=events)
f = benford(list(df.groupby(pd.Grouper(freq='1D')).count()[0]),f'all lightning in 1D')
# %% uniform
scale = 2 * mean
events = []
temp = start
for i in range(1000000):
    if temp < end:
        events.append(temp)
    else:
        break
    temp = temp + timedelta(minutes=scale * np.random.uniform())

col = [1 for i in range(len(events))]
df = pd.DataFrame(col,index=events)
f = benford(list(df.groupby(pd.Grouper(freq='1D')).count()[0]),f'all lightning in 1D')
