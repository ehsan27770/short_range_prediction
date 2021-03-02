#%%
import numpy as np
import pandas as pd
from benford import benford
import matplotlib.pyplot as plt
#%%
dfs = [pd.read_csv(r"./rawData/order{}.txt".format(i),sep=';',low_memory=False) for i in range(2008,2019)]

#data_outer = pd.concat(dfs,join='outer',ignore_index=True)
data_inner = pd.concat(dfs,join='inner',ignore_index=True)
data_inner


data = data_inner.copy()
data
#%%

to_drop = ['stn','nto000s0','hto000s0','vho000sw','w3p000iw','w5p000iw','w1p000iw','w2p000iw','wat000sw','nto000sw','cha000s0']
data = data.drop(to_drop,axis = 1)
data
#%%
for col in list(data.columns)[1:]:
    data[col] = pd.to_numeric(data[col],errors='coerce')

data = data.dropna()
data = data.reset_index(drop=True)
# %%
data['time'] = pd.to_datetime(data.time.astype('str'),format='%Y%m%d%H%M')
data = data.set_index(['time'])
# %%
data['bre0'] = data['brefarz0'] + data['brecloz0']
data['onoff'] = np.sign(data['bre0'])
delivery_add = pd.DataFrame([['bre0','lightning count','No'],['onoff','lightning onoff','No']],columns=['id','name','unit'])
delivery_add = delivery_add.set_index(['id'])
delivery_add
# %%

data

# %%
data.to_csv('./cleanData/all.csv')






# %% start
data = pd.read_csv('./cleanData/all.csv',index_col='time',parse_dates=['time'],infer_datetime_format=True)
delivery_note = pd.read_csv('./rawData/delivery_note.csv',sep='\t')
delivery_note = delivery_note.set_index(['id'])
delivery_note = delivery_note.append(delivery_add)
data
# %%


for col in data.columns:
    benford(data[col],delivery_note.loc[col]['name']+'('+delivery_note.loc[col]['unit']+')')


# %% training
x = data.drop(columns=['brecloz0','brefarz0','onoff'])
y = x.pop('bre0')

msk = np.random.rand(len(x)) < 0.8
x_train = x[msk]
x_test = x[~msk]
y_train = y[msk]
y_test = y[~msk]

data


# %% Linear
from sklearn import linear_model
linear = linear_model.LinearRegression()
linear.fit(x_train,y_train)
linear.score(x_train,y_train)

print('coefficient: \n',linear.coef_)
print('Intercept: \n',linear.intercept_)

predicted = linear.predict(x_test)
predicted
y_test.values

linear.score(x_test,y_test)


# %% logestic regression
from sklearn import linear_model
linear = linear_model.LogisticRegression()
linear.fit(x_train,y_train)
linear.score(x_train,y_train)

print('coefficient: \n',linear.coef_)
print('Intercept: \n',linear.intercept_)

predicted = linear.predict(x_test)
predicted
y_test.values

linear.score(x_test,y_test)

# %% decision tree
from sklearn import tree
model = tree.DecisionTreeRegressor()
model.fit(x_train,y_train)
model.score(x_train,y_train)
model.score(x_test,y_test)
predicted = model.predict(x_test)
predicted
y_test.values
