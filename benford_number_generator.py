import numpy as np
import matplotlib.pyplot as plt
from benford import benford, normalize
import math
import random
from mpl_toolkits.mplot3d import axes3d
# %%
def benfords_range_gen(stop, n):
    """ A generator that returns n random integers
    between 1 and stop-1 and whose distribution
    meets Benford's Law i.e. is logarithmic.
    """
    multiplier = math.log(stop)
    for i in range(n):
        #yield int(math.exp(multiplier * random.random()))
        yield math.exp(multiplier * random.random())
# %%
max = 1000000
count = 20000
x = benfords_range_gen(max,count)
a = []
for i in x:
    a.append(i)
# %%
benford(a,'initial distribution')
# %%
perturbed = []
rhos = np.arange(0.5,1.5,0.01)
for rho in rhos:
    jsd, kld = normalize(a,rho)
    perturbed.append(jsd)

plt.figure(figsize=(15,10))
plt.plot(rhos, perturbed)
#plt.xscale('log')

# %%
perturbed = []
rhos = np.arange(0.08,11,0.01)
for rho in rhos:
    jsd, kld = normalize(a,rho)
    perturbed.append(jsd)

plt.figure(figsize=(15,10))
plt.plot(rhos, perturbed)
plt.xscale('log')
# %%
rhos = np.arange(0.5,1.5,0.1)
for rho in rhos:
    b = []
    for i in a:
        b.append(i*rho)
    benford(b,f'rho = {rho}')
benford(a,'rho = 1')


# %%

for rho in [1,2,3,4,5,6,7,8,9,10,11,12]:
    b = []
    for i in a:
        b.append(i*rho)
    benford(b,f'rho = {rho}')
# %%


maxs = np.logspace(2,6,50)
counts = np.logspace(2,6,50)

X, Y = np.meshgrid(maxs,counts,indexing='xy')
divergence = np.empty([maxs.size,counts.size])
for i,max in enumerate(maxs):
    for j,count in enumerate(counts):
        x = benfords_range_gen(int(max),int(count))
        jsd,kld = normalize(x)
        divergence[j,i] = jsd

# %%
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(np.log10(X),np.log10(Y),divergence)
ax.set_xlabel('maximum number')
ax.set_ylabel('number of samples')
ax.view_init(10, 30)
# %%
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(np.log10(X),np.log10(Y),divergence)
ax.set_xlabel('maximum number')
ax.set_ylabel('number of samples')
ax.view_init(10, 70)
# %%
