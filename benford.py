# %% codecell
import numpy as np
import os
import matplotlib.pyplot as plt
from math import log10,log
import scipy.stats


def counter(numbers,rho=None,limit=None):
    dist = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    count = 0
    numbers_modified = []
    for num in numbers:
        if rho:
            num = int(num * rho)
        if limit:
            if num > limit:
                continue
        try:
            digit = int(str(num)[0])
            if digit < 1:
                continue
            dist[digit] = dist[digit] + 1
            count = count + 1
            numbers_modified.append(num)
        except:
            pass

    x = list(range(1,10))
    y = [dist[i]/count for i in x]
    t = [log10(1+1.0/i) for i in x]

    x = np.array(x)
    y = np.array(y)
    t = np.array(t)

    return x, y, t, min(numbers_modified), max(numbers_modified), len(numbers_modified)

def benford(numbers,title,limit=None,ax=None,rho=None):
    x, y, t, inf, sup, num= counter(numbers,limit=limit,rho=rho)
    if ax is None:
        f, ax = plt.subplots(figsize = (10,5))

    l1 = ax.bar(x,y,label='real data')

    l2, = ax.plot(x,t,'r-o',label='ideal benford')
    ax.legend(handles=[l1,l2], loc='best')

    jsd = jensen_shannon_distance(y,t)

    kld = Kullback_Leibler_divergence(y,t)
    ax.title.set_text(title + f'\nJSD={jsd:.3f}\nrange: {inf}-{sup}')
    #ax.title.set_text(title + f'\nJSD={np.exp(jsd):.3f}\nrange: {inf}-{sup}')
    return


    #mse = np.mean(np.square(y-t))
    #mae = np.mean(np.abs(y-t))
    #ksd = Kolmogorov_Smirnov_distance(y,t)
    #csd = Chi_square_distance(y,t)
    #bd = Bhattacharyya_distance(y,t)
    #hd = Hellinger_distance(y,t)
    #l = lambda_distance(y,t,0.5)
    #plt.title(title + '\n\nmse={}\nmae={},\nJSD={}\nKSD={}\nKLD={}\ncsd={}\nBD={}\nHD={}\nlambda={}'.format(mse,mae,jsd,ksd,kld,csd,bd,hd,l))


def distance(numbers,rho=None,limit=None):#previous name was normalize
    x, y, t, inf, sup, num = counter(numbers,rho=rho,limit=limit)
    jsd = jensen_shannon_distance(y,t)
    kld = Kullback_Leibler_divergence(y,t)

    return jsd, kld, inf, sup, num


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
    distance = np.sqrt(divergence)

    return distance

def Kolmogorov_Smirnov_distance(p,q):
    p = p.cumsum()
    q = q.cumsum()

    distance = np.max(np.abs(p-q))
    return distance

def Kullback_Leibler_divergence(p,q):
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
if __name__ == "__main__":
    plt.rcParams['font.size'] = 12
    x = list(range(1,10))
    y = [log10(1+1.0/i) for i in x]
    f, ax = plt.subplots(figsize = (10,5))


    #ax.legend(handles=[l1], loc='best')
    for i, v in enumerate(y):
        ax.text(i + 0.650, v + .003, f'{100*v:.2f}%', color='blue', fontweight='bold')
    l1 = ax.bar(x,y,label='real data')
    plt.rcParams['font.size'] = 15
    plt.xticks(x)
    plt.xlabel('first digit')
    ax.set_ylabel('probability')
    plt.title('Benford Law distribution')
    plt.savefig('benford.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)
