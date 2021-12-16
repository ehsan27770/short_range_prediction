import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.image import imread
from mpl_toolkits.axisartist.axislines import SubplotZero
import numpy as np

from cartopy import config
import cartopy.crs as ccrs
import networkx as nx
# %%
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################

fig = plt.figure(figsize=(20,14))

desired_projections = [ccrs.PlateCarree()]
for plot_num, desired_proj in enumerate(desired_projections):

    ax = plt.subplot(1, 1, plot_num + 1, projection=desired_proj)


    ax.set_global()

    ax.add_patch(mpatches.Rectangle(xy=[11, 46], width=6, height=3.5,facecolor='None',edgecolor='red',linewidth=3,alpha=0.9,transform=ccrs.PlateCarree()))
    ax.set_extent([0,30,35,55],desired_proj)


    #fname = os.path.join(config["repo_data_dir"],'raster', 'natural_earth', '50-natural-earth-1-downsampled.png')
    #fname = os.path.join(config["repo_data_dir"],'raster', 'sample', 'Miriam.A2012270.2050.2km.jpg')
    #img = imread(fname)
    #ax.imshow(img, origin='upper', transform=ccrs.PlateCarree(), extent=[-180, 180, -90, 90])
    #ax.stock_img()
    ax.background_img("natural-earth-1",resolution='large')

    ax.gridlines()
    ax.coastlines()

plt.show()

fig.savefig(f'saved_images_new_range/world.png',dpi=300,bbox_inches='tight')
# %%
plt.figure(figsize=(20,14))

desired_projections = [ccrs.PlateCarree(),ccrs.RotatedPole(pole_latitude=45, pole_longitude=180)]
for plot_num, desired_proj in enumerate(desired_projections):

    ax = plt.subplot(2, 1, plot_num + 1, projection=desired_proj)
    #ax.set_extent([5,30,40,60], desired_proj)

    ax.set_global()

    #ax.add_patch(mpatches.Rectangle(xy=[13.5, 46.8], width=2.5, height=1.7,facecolor='blue',alpha=0.2,transform=ccrs.PlateCarree()))
    print(ax.get_extent())
    ax.set_extent([-90,40,-60,60],desired_proj)
    print(ax.get_extent())
    ax.add_patch(mpatches.Rectangle(xy=[-70, -45], width=90, height=90,facecolor='blue',alpha=0.2,transform=ccrs.PlateCarree()))
    #ax.add_patch(mpatches.Rectangle(xy=[70, -45], width=90, height=90,facecolor='red',alpha=0.2,transform=ccrs.Geodetic()))

    ax.gridlines()
    ax.coastlines()

plt.show()


# %%
proj = ccrs.AlbersEqualArea(central_longitude=-35,
                            central_latitude=40,
                            standard_parallels=(0, 80))
ax = plt.axes(projection=proj)
ax.set_extent([-100, 30, 0, 80], crs=ccrs.PlateCarree())
ax.coastlines()

# Make a boundary path in PlateCarree projection, I choose to start in
# the bottom left and go round anticlockwise, creating a boundary point
# every 1 degree so that the result is smooth:
vertices = [(lon, 0) for lon in range(-100, 31, 1)] + [(lon, 80) for lon in range(30, -101, -1)]
boundary = mpath.Path(vertices)

ax.set_boundary(boundary, transform=ccrs.PlateCarree())

plt.show()

# %%
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
x = np.linspace(-np.pi, np.pi, 100)
y = 2 * np.sin(x)

rc = {"xtick.direction" : "inout", "ytick.direction" : "inout",
      "xtick.major.size" : 5, "ytick.major.size" : 5,}
with plt.rc_context(rc):
    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # make arrows
    ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
            transform=ax.get_xaxis_transform(), clip_on=False)

    plt.show()



# %%
x = np.logspace(0, 4, 100)
y = np.zeros_like(x)

rc = {"xtick.direction" : "inout", "ytick.direction" : "inout",
      "xtick.major.size" : 30, "ytick.major.size" : 30,
      "xtick.minor.size" : 15, "ytick.minor.size" : 15, "font.size" : 20}
with plt.rc_context(rc):
    fig, ax = plt.subplots(figsize=(25,4))
    ax.plot(x, y, alpha = 0)

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # make arrows
    ax.plot((1), (0), ls="", marker=">", ms=10, color="k",transform=ax.get_yaxis_transform(), clip_on=False)
    #ax.plot((0), (1), ls="", marker="^", ms=10, color="k",transform=ax.get_xaxis_transform(), clip_on=False)

    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlim([0,11000])
    #ax.set_xscale('log')

    plt.show()
    fig.savefig('saved_images/axis_lin.png',dpi=300,bbox_inches='tight')


# %%
x = np.logspace(0, 4, 100)
y = np.zeros_like(x)

rc = {"xtick.direction" : "inout", "ytick.direction" : "inout",
      "xtick.major.size" : 30, "ytick.major.size" : 30,
      "xtick.minor.size" : 15, "ytick.minor.size" : 15, "font.size" : 20}
with plt.rc_context(rc):
    fig, ax = plt.subplots(figsize=(25,4))
    ax.plot(x, y, alpha = 0)

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # make arrows
    ax.plot((1), (0), ls="", marker=">", ms=10, color="k",transform=ax.get_yaxis_transform(), clip_on=False)
    #ax.plot((0), (1), ls="", marker="^", ms=10, color="k",transform=ax.get_xaxis_transform(), clip_on=False)

    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlim([0.11,19000])
    ax.set_xscale('log')

    plt.show()
    fig.savefig('saved_images/axis_log.png',dpi=300,bbox_inches='tight')

#plt.rcParams
# %%
x = np.logspace(0, 4, 100)
y = np.zeros_like(x)

rc = {"xtick.direction" : "inout", "ytick.direction" : "inout",
      "xtick.major.size" : 30, "ytick.major.size" : 30,
      "xtick.minor.size" : 15, "ytick.minor.size" : 15, "font.size" : 20}
with plt.rc_context(rc):
    fig, ax = plt.subplots(figsize=(25,4))
    ax.plot(x, y, alpha = 0)

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # make arrows
    ax.plot((1), (0), ls="", marker=">", ms=10, color="k",transform=ax.get_yaxis_transform(), clip_on=False)
    #ax.plot((0), (1), ls="", marker="^", ms=10, color="k",transform=ax.get_xaxis_transform(), clip_on=False)

    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)


    ax.set_xlim([0.095,110])
    ax.set_xscale('log')
    pos = [i/10 for i in range(1,10)] + [i for i in range(1,10)] + [10*i for i in range(1,11)]
    ax.set_xticks(pos)
    ax.set_xticklabels(pos,rotation=90)
    #ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().get_major_formatter().labelOnlyBase = False

    plt.show()
    fig.savefig('saved_images/axis_log_report.png',dpi=300,bbox_inches='tight')


# %%
#x = np.logspace(0, 4, 100)
#y = np.zeros_like(x)

rc = {"xtick.direction" : "inout", "ytick.direction" : "inout",
      "xtick.major.size" : 30, "ytick.major.size" : 30,
      "xtick.minor.size" : 15, "ytick.minor.size" : 15, "font.size" : 20}
with plt.rc_context(rc):
    fig, ax = plt.subplots(figsize=(10,8))
    #ax.plot(x, y, alpha = 0)

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # make arrows
    ax.plot((1), (0), ls="", marker=">", ms=10, color="k",transform=ax.get_yaxis_transform(), clip_on=False)
    #ax.plot((0), (1), ls="", marker="^", ms=10, color="k",transform=ax.get_xaxis_transform(), clip_on=False)

    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlim([0,10**5*1.035])
    #ax.set_ylim([-5,1])
    #ax.set_xscale('log')


    ax2 = ax.twiny()
    ax2.spines['left'].set_position('zero')
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_position(('data',-1))
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')

    # make arrows
    ax.plot((1), (-1), ls="", marker=">", ms=10, color="k",transform=ax.get_yaxis_transform(), clip_on=False)
    #ax.plot((0), (1), ls="", marker="^", ms=10, color="k",transform=ax.get_xaxis_transform(), clip_on=False)

    ax2.yaxis.set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_xlim([10**1,10**5*1.4])
    #ax2.set_ylim([-6,1])
    ax2.set_xscale('log')




    x_total = np.empty(4 * 100)
    for i in range(4):
        p = ax.twiny()
        p.spines['top'].set_visible(False)
        p.spines['left'].set_visible(False)
        p.spines['right'].set_visible(False)
        p.spines['bottom'].set_position(('data',-i-1-1))
        p.xaxis.set_ticks_position('bottom')
        p.yaxis.set_visible(False)
        #p.set_ylim([-5,1])
        p.set_xscale('log')
        #p.set_xlim([0.12,19000])
        #ax.plot((1), (0-0.115*i), ls="", marker=">", ms=10, color="k",transform=p.get_yaxis_transform(), clip_on=False)
        ax.plot((1), (0-i-1-1), ls="", marker=">", ms=10, color="k",transform=ax.get_yaxis_transform(), clip_on=False)
        #ax.plot((1), (0-i), ls="", marker=">", ms=10, color="k", clip_on=False)
        x = np.power(10,i + 1 + np.random.rand(100))
        y = np.zeros_like(x)
        p.plot(x,y-i-1-1,'o',alpha=0.5)
        x_total[i*100:(i+1)*100] = x
        p.set_xlim([10**(i+1),10**(i+2)*1.1])



    y_total = np.zeros_like(x_total)
    ax.plot(x_total,y_total,'o',alpha=0.5)
    ax2.plot(x_total,y_total-1,'o',alpha=0.5)

    plt.show()
    fig.savefig('saved_images/axis_log_grid.png',dpi=300,bbox_inches='tight')
# %%

import collections
Input = collections.namedtuple('Input', ['start', 'end', 'word_list'])
test_input = Input('swiss', 'wines', ['chaps', 'chats', 'chips', 'coats', 'costs', 'lines', 'lives', 'loses', 'loves', 'poses', 'posts', 'ships', 'skims', 'skips', 'swims', 'swiss', 'wanes', 'wines'])
word_list = ['chaps', 'chats', 'chips', 'coats', 'costs', 'lines', 'lives', 'loses', 'loves', 'poses', 'posts', 'ships', 'skims', 'skips', 'swims', 'swiss', 'wanes', 'wines']
word_list = ['dag', 'pee', 'red', 'rex', 'tad', 'tag', 'tax', 'ted', 'tex']
is_neighbor = lambda x, y: sum(a != b for a, b in zip(x, y)) == 1

dod = {}
for node in word_list:
    neighbor_words = [w for w in word_list if is_neighbor(node, w)]
    dod[node] = {w: {'weight': 1} for w in neighbor_words}
dod
G = nx.Graph(dod)
#help(G)
#nx.draw(G,with_labels=True,with_weights=True)

#pos=nx.get_node_attributes(G,'pos')
pos=nx.spring_layout(G)
nx.draw(G,pos,with_labels=True)
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
