# %% codecell
import numpy as np
import os
import matplotlib.pyplot as plt
from math import log10


# %%
def benford(numbers,title):
    dist = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    count = 0
    for num in numbers:
        try:
            digit = int(str(num)[0])
            if digit < 1:
                continue
            dist[digit] = dist[digit] + 1
            count = count + 1
        except:
            pass

    x = list(range(1,10))
    y = [dist[i]/count for i in x]
    t = [log10(1+1.0/i) for i in x]

    f = plt.figure()
    plt.bar(x,y)
    #plt.plot(x,y)
    plt.plot(x,t,'r')
    plt.title(title)
    #return f
# %%
