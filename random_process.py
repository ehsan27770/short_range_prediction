import numpy as np
import pandas as pd
from datetime import datetime,timedelta
import random
from benford import benford,distance
import matplotlib.pyplot as plt


# %%
start = datetime(year=2021,month=1,day=1,hour=0,minute=0,second=0,microsecond=0)
end = datetime(year = 2026,month=1,day=1,hour=0,minute=0,second=0,microsecond=0)
#np.random.binomial()
#np.random.poisson()
#np.random.rand()

# %%
mean_tunder = 7 #day
mean_duration = 12 #hour
mean_light = 1 #min

# %% test
def random_number_generator(type):
    #@functools.wraps(func)
    def rayleigh(mean):
        return np.random.rayleigh(scale = mean/1.253)
    def exponential(mean):
        return np.random.exponential(scale = mean)
    def uniform(mean):
        return 2* mean * np.random.uniform()
    if type == 1:
        return rayleigh,"rayleigh"
    elif type == 2:
        return exponential,"exponential"
    elif type == 3:
        return uniform,"uniform"

# %%
dict = {}
# %%
random_number, distribution = random_number_generator(type = 3)
events = []
next_tunder_event_start = start
while next_tunder_event_start < end:
    end_of_tunder = next_tunder_event_start + timedelta(hours = random_number(mean_duration))
    temp = next_tunder_event_start
    while temp < end_of_tunder:
        events.append(temp)
        temp = temp + timedelta(minutes = random_number(mean_light))
    next_tunder_event_start = end_of_tunder + timedelta(days = random_number(mean_tunder))

col = [1 for i in range(len(events))]
df = pd.DataFrame(col,index=events)
df
# %%
for i in [1,6,12,24,36,48,60,100,600,1200]:
    f = benford(list(df.groupby(pd.Grouper(freq=f'{i}H')).count()[0]),f'all lightning in {i}H with interarrival time from {distribution} distribution')
    #f.savefig(f'simulation_groupby_{i}_H_{distribution}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)

# %%
x = np.logspace(1,3,100).astype('uint')
y = []
sups = []
infs = []
for i in x:
    jsd, kld, inf, sup, num = distance(list(df.groupby(pd.Grouper(freq=f'{i}H')).count()[0]))
    y.append(jsd)
    sups.append(sup)
    infs.append(inf)
fig, ax = plt.subplots(figsize = (10,5))
part1 = ax.twinx()
part2 = ax.twinx()

p0, = ax.plot(x,y,label='JSD')
ax.set_xlabel('time slice (hour)')
ax.set_ylabel('jensen shannon distance')

p1, = part1.plot(x,sups,'r',label='maximum')
part1.set_ylabel('maximum')

p2, = part2.plot(x,infs,'g',label='minimum')
part2.set_ylabel('minimum')
#plt.xticks(x)
plt.xscale('log')
lns = [p0,p1,p2]
ax.legend(handles=lns, loc='best')


# right, left, top, bottom
part2.spines['right'].set_position(('outward', 60))

# no x-ticks
#part2.xaxis.set_ticks([])

# Sometimes handy, same for xaxis
#part2.yaxis.set_ticks_position('right')

# Move "Velocity"-axis to the left
# part2.spines['left'].set_position(('outward', 60))
# part2.spines['left'].set_visible(True)
# part2.yaxis.set_label_position('left')
# part2.yaxis.set_ticks_position('left')

ax.yaxis.label.set_color(p0.get_color())
part1.yaxis.label.set_color(p1.get_color())
part2.yaxis.label.set_color(p2.get_color())
fig.tight_layout()
plt.title(f"simulation with {distribution} interarrival time")
fig.savefig(f'simulation_{distribution}.png',dpi=fig.dpi,bbox_inches='tight',pad_inches=.5)
dict[f'{distribution}'] = [x,y,sups,infs]

# %%
dict

# %%
fig, ax = plt.subplots(figsize = (10,5))
lns = []
for name in ['rayleigh','exponential','uniform']:
    ln, = ax.plot(dict[name][0],dict[name][1],label=name)
    lns.append(ln)
ax.legend(handles=lns, loc='best')
plt.xscale('log')
plt.title('jensen-shannon distance with respect to different time slices')
ax.set_xlabel('time slice (hour)')
ax.set_ylabel('jensen-shannon distance')
fig.savefig(f'comparison.png',dpi=10*fig.dpi,bbox_inches='tight',pad_inches=.5)
# %% rayleigh https://en.wikipedia.org/wiki/Rayleigh_distribution


# %% exponential https://en.wikipedia.org/wiki/Exponential_distribution

# %% uniform
