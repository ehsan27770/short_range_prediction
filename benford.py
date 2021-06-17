# %% codecell
import numpy as np
import os
import matplotlib.pyplot as plt
from math import log10,log
import scipy.stats


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

    x = np.array(x)
    y = np.array(y)
    t = np.array(t)

    f, ax = plt.subplots(figsize = (10,5))
    plt.bar(x,y)
    #plt.plot(x,y)
    plt.plot(x,t,'r')
    mse = np.mean(np.square(y-t))
    mae = np.mean(np.abs(y-t))
    jsd = jensen_shannon_distance(y,t)
    ksd = Kolmogorov_Smirnov_distance(y,t)
    kld = Kullback_Leibler_distance(y,t)
    csd = Chi_square_distance(y,t)
    bd = Bhattacharyya_distance(y,t)
    hd = Hellinger_distance(y,t)
    l = lambda_distance(y,t,0.5)
    #plt.title(title + '\n\nmse={}\nmae={},\nJSD={}\nKSD={}\nKLD={}\ncsd={}\nBD={}\nHD={}\nlambda={}'.format(mse,mae,jsd,ksd,kld,csd,bd,hd,l))
    plt.title(title + f'\n\nJSD={jsd}')
    return

def normalize(numbers,rho=None):
    dist = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    count = 0
    for num in numbers:
        if rho:
            num = int(num * rho)
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

    x = np.array(x)
    y = np.array(y)

    jsd = jensen_shannon_distance(y,t)
    kld = Kullback_Leibler_distance(y,t)

    return jsd, kld
# %%



def jensen_shannon_distance(p,q):
    """
    method to compute the Jenson-Shannon Distance
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    #distance = np.sqrt(divergence)

    return divergence

def Kolmogorov_Smirnov_distance(p,q):
    p = p.cumsum()
    q = q.cumsum()

    distance = np.max(np.abs(p-q))
    return distance

def Kullback_Leibler_distance(p,q):
    #divergence = scipy.stats.entropy(p, q) + scipy.stats.entropy(q, p)
    divergence = scipy.stats.entropy(p, q)
    #distance = np.sqrt(divergence)

    return divergence

def lambda_distance(p,q,ld):

    divergence = ld*scipy.stats.entropy(p, ld*p+(1-ld)*q) + (1-ld)*scipy.stats.entropy(q, (1-ld)*p+ld*q)
    distance = np.sqrt(divergence)

    return distance

def Bhattacharyya_distance(p,q):
    BC = np.sum(np.sqrt(p*q))
    distance = -log(BC)
    return distance

def Hellinger_distance(p,q):
    BC = np.sum(np.sqrt(p*q))
    distance = np.sqrt(1-BC)
    return distance

def Chi_square_distance(p,q):
    distance = 0.5 * np.sum(np.square(p-q)/(p+q))
    return distance

# %%
