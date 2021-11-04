import numpy as np
import matplotlib.pyplot as plt
from benford import benford, distance
import math
import random
from mpl_toolkits.mplot3d import axes3d
import time
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
fig, ax = plt.subplots(figsize=(25,14))
benford(a,'Initial Distribution',ax=ax)
ax.set_xticks(range(1,10))
ax.set_xlabel("first digit")
ax.set_ylabel("probability")
fig.savefig(f'saved_images/ideal_benford.png',dpi=300,bbox_inches='tight')
# %%
perturbed = []
rhos = np.arange(0.5,1.5,0.01)
for rho in rhos:
    jsd, kld, inf, sup, num = distance(a,rho)
    perturbed.append(jsd)

plt.figure(figsize=(15,10))
plt.plot(rhos, perturbed)
#plt.xscale('log')

# %%
perturbed = []
rhos = np.arange(0.08,11,0.01)
for rho in rhos:
    jsd, kld, inf, sup, num = distance(a,rho)
    perturbed.append(jsd)

f, ax = plt.subplots(figsize=(15,10))
plt.plot(rhos, perturbed)
plt.xscale('log')
plt.xlabel("multiplier")
plt.ylabel("JSD")
#f.savefig(f'multiply.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)


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
        gen = benfords_range_gen(int(max),int(count))
        x = []
        for g in gen:
            x.append(g)
        jsd, kld, inf, sup, num = distance(x)
        divergence[j,i] = jsd

# %%
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(np.log10(X),np.log10(Y),divergence)
ax.set_xlabel('\nmaximum number')
ax.set_ylabel('\nnumber of samples(time slices)')
ax.set_zlabel('JSD')
ax.xaxis.set_ticks([i for i in range(2,7)])
ax.yaxis.set_ticks([i for i in range(2,7)])

ax.xaxis.set_ticklabels([f'$10^{i}$' for i in range(2,7)],fontsize=15)
ax.yaxis.set_ticklabels([f'$10^{i}$' for i in range(2,7)],fontsize=15)




ax.view_init(10, 30)
fig.savefig(f'saved_images/3D_1.png',dpi=300,bbox_inches='tight')
# %%
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(np.log10(X),np.log10(Y),divergence)
ax.set_xlabel('\nmaximum number')
ax.set_ylabel('\nnumber of samples(time slices)')
ax.set_zlabel('JSD')
ax.xaxis.set_ticks([i for i in range(2,7)])
ax.yaxis.set_ticks([i for i in range(2,7)])

ax.xaxis.set_ticklabels([f'$10^{i}$' for i in range(2,7)],fontsize=15)
ax.yaxis.set_ticklabels([f'$10^{i}$' for i in range(2,7)],fontsize=15)

ax.view_init(10, 70)
fig.savefig(f'saved_images/3D_2.png',dpi=300,bbox_inches='tight')
# %%
fig,ax = plt.subplots(figsize=(15,10))
ax.plot(np.log10(X),divergence,'C0')
ax.set_ylabel('JSD')
ax.set_xlabel('maximum number')
ax.xaxis.set_ticks([i for i in range(2,7)])
ax.xaxis.set_ticklabels([f'$10^{i}$' for i in range(2,7)])
fig.savefig(f'saved_images/side_X.png',dpi=300,bbox_inches='tight')

# %%
fig,ax = plt.subplots(figsize=(15,10))
ax.plot(np.log10(Y),divergence,'C0')
ax.set_ylabel('JSD')
ax.set_xlabel('number of samples(time slices)')
ax.xaxis.set_ticks([i for i in range(2,7)])
ax.xaxis.set_ticklabels([f'$10^{i}$' for i in range(2,7)])
fig.savefig(f'saved_images/side_Y.png',dpi=300,bbox_inches='tight')

# %%


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

maxs = np.logspace(2,8,7)
maxs = np.array([10**4,])
counts = np.logspace(2,8,7)

X, Y = np.meshgrid(maxs,counts,indexing='xy')
divergence = np.empty([counts.size,maxs.size])
for i,max in enumerate(maxs):
    for j,count in enumerate(counts):
        t0 = time.time()
        gen = benfords_range_gen(int(max),int(count))
        x = []
        for g in gen:
            x.append(g)

        t1 = time.time()
        print(f'step1 = {t1-t0}')
        jsd, kld, inf, sup, num = distance(x)
        t2 = time.time()
        print(f'step2 = {t2-t1}')
        divergence[j,i] = jsd

# %%
fig,ax = plt.subplots(figsize=(15,10))
ax.plot(np.log10(Y),divergence)
ax.set_ylabel('JSD')
ax.set_xlabel('number of samples(time slices)')
ax.xaxis.set_ticks([i for i in range(2,9)])
ax.xaxis.set_ticklabels([f'$10^{i}$' for i in range(2,9)])
fig.savefig(f'saved_images/sweep.png',dpi=300,bbox_inches='tight')

# %%
maxs = np.logspace(2,8,7)
#maxs = np.array([10**4,])
counts = np.logspace(2,8,7)

X, Y = np.meshgrid(maxs,counts,indexing='xy')
divergence = np.empty([counts.size,maxs.size])
for i,max in enumerate(maxs):
    for j,count in enumerate(counts):
        t0 = time.time()
        gen = benfords_range_gen(int(max),int(count))
        x = []
        for g in gen:
            x.append(g)

        t1 = time.time()
        print(f'step1 = {t1-t0}')
        jsd, kld, inf, sup, num = distance(x)
        t2 = time.time()
        print(f'step2 = {t2-t1}')
        divergence[j,i] = jsd

# %%
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(np.log10(X),np.log10(Y),divergence)
ax.set_xlabel('\nmaximum number')
ax.set_ylabel('\nnumber of samples(time slices)')
ax.set_zlabel('JSD')
ax.xaxis.set_ticks([i for i in range(2,9)])
ax.yaxis.set_ticks([i for i in range(2,9)])

ax.xaxis.set_ticklabels([f'$10^{i}$' for i in range(2,9)],fontsize=15)
ax.yaxis.set_ticklabels([f'$10^{i}$' for i in range(2,9)],fontsize=15)

ax.view_init(10, 30)
fig.savefig(f'saved_images/3D_only10_1.png',dpi=300,bbox_inches='tight')
# %%
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(np.log10(X),np.log10(Y),divergence)
ax.set_xlabel('\nmaximum number')
ax.set_ylabel('\nnumber of samples(time slices)')
ax.set_zlabel('JSD')
ax.xaxis.set_ticks([i for i in range(2,9)])
ax.yaxis.set_ticks([i for i in range(2,9)])

ax.xaxis.set_ticklabels([f'$10^{i}$' for i in range(2,9)],fontsize=15)
ax.yaxis.set_ticklabels([f'$10^{i}$' for i in range(2,9)],fontsize=15)

ax.view_init(10, 70)
fig.savefig(f'saved_images/3D_only10_2.png',dpi=300,bbox_inches='tight')
