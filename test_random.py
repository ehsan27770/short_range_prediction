import hashlib
import urllib
import time

#'https://coronavax.unisante.ch/'

local_data = urllib.request.urlopen('https://www.google.com/finance/').read()
local_hash = hashlib.md5(remote_data).hexdigest()


for i in range(50):
    remote_data = urllib.request.urlopen('https://www.google.com/finance/').read()
    remote_hash = hashlib.md5(remote_data).hexdigest()
    if remote_hash == local_hash:
        print('no changed')
    else:
        print('warning')
    local_hash = remote_hash
    time.sleep(1)
# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benford import benford, distance

df = pd.read_csv('data/08. Lightning Location Data of Suzhou 2021.csv',index_col='Time',parse_dates=['Time'])
df
# %%
fig, ax = plt.subplots(figsize=(15,10))
benford(list(df[df['Amplitude']<0]['Amplitude'].groupby(pd.Grouper(freq='1D')).count()),'Negative',ax=ax)
fig.savefig('negative.png',dpi=300)
# %%
fig, ax = plt.subplots(figsize=(15,10))
benford(list(df[df['Amplitude']>0]['Amplitude'].groupby(pd.Grouper(freq='1D')).count()),'Positive',ax=ax)
fig.savefig('positive.png',dpi=300)
