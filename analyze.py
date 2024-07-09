# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from benford import benford, distance
import sys
sys.version
plt.rcParams['font.size'] = 15
from scipy.interpolate import make_interp_spline, BSpline

import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


find_pos = lambda x : np.abs(x) if x>0 else np.nan
find_neg = lambda x : np.abs(x) if x<0 else np.nan
from scipy.stats import wasserstein_distance

# %%
#-------------------------------------------------------------------------------
source = 'stroke'
# %%
#-------------------------------------------------------------------------------
dfs = [pd.read_csv(f'data/data_2000-2004_Impact_System_new_region/{source}data_200{i}.asc',sep='|',names=['num','date','nano','st_y','st_x','amplitude','icloud','flash'],index_col='date',parse_dates=['date']) for i in range(5)]
df1 = pd.concat(dfs,join='inner',ignore_index=False)
df1['amp_abs'] = np.abs(df1['amplitude'])
df1['amp_pos'] = df1['amplitude'].apply(find_pos)
df1['amp_neg'] = df1['amplitude'].apply(find_neg)
print(df1)

dfs = [pd.read_csv(f'data/data_2010-2014_LS7000_System_new_region/{source}data_201{i}.asc',sep='|',names=['num','date','nano','st_y','st_x','amplitude','icloud','flash'],index_col='date',parse_dates=['date']) for i in range(5)]
df2 = pd.concat(dfs,join='inner',ignore_index=False)
df2['amp_abs'] = np.abs(df2['amplitude'])
df2['amp_pos'] = df2['amplitude'].apply(find_pos)
df2['amp_neg'] = df2['amplitude'].apply(find_neg)
print(df2)

dfs = [pd.read_csv(f'data/data_2016-2020_LS7002_System_new_region/{source}data_20{i}.asc',sep='|',names=['num','date','nano','st_y','st_x','amplitude','icloud','flash'],index_col='date',parse_dates=['date']) for i in range(16,21)]
df3 = pd.concat(dfs,join='inner',ignore_index=False)
df3['amp_abs'] = np.abs(df3['amplitude'])
df3['amp_pos'] = df3['amplitude'].apply(find_pos)
df3['amp_neg'] = df3['amplitude'].apply(find_neg)
print(df3)



# %%
#-------------------------------------------------------------------------------
#general statistics
vals = np.empty((3,2,2))
vals.shape
for i,df in enumerate([df1,df2,df3]):
    l1 = len(df)
    df = df[df['icloud']=='f']
    l2 = len(df)
    lp = len(df[df['amplitude']>0])
    ln = len(df[df['amplitude']<0])
    #print(l1,l2,lp,ln)
    vals[i,0,0] = lp
    vals[i,0,1] = ln
    #print(f'total:{l1}(100\%), cloud:{l1-l2}({(l1-l2)/l1*100:.3}\%), ground:{l2}({l2/l1*100:.3}\%),pos:{lp}({lp/l2*100:.3}\%),neg:{ln}({ln/l2*100:.3}\%)')
    #print(f' & {l1}(100\%) & {l1-l2}({(l1-l2)/l1*100:.3}\%) & {l2}({l2/l1*100:.3}\%) & {l2}(100\%) & {lp}({lp/l2*100:.3}\%) & {ln}({ln/l2*100:.3}\%) \\\\')
    print(f' & {l1} & {l1-l2} & {l2} & {l2} & {lp} & {ln} \\\\')
    print(f' & (100\%) & ({(l1-l2)/l1*100:.3}\%) & ({l2/l1*100:.3}\%) & (100\%) & ({lp/l2*100:.3}\%) & ({ln/l2*100:.3}\%) \\\\ \n\hline')

for i,df in enumerate([df1,df2,df3]):
    l1 = len(df)
    df = df[df['icloud']=='t']
    l2 = len(df)
    lp = len(df[df['amplitude']>0])
    ln = len(df[df['amplitude']<0])
    print(l1,l2,lp,ln)
    vals[i,1,0] = lp
    vals[i,1,1] = ln
# %%
df1.info()
# %%
#-------------------------------------------------------------------------------
size = 0.5
fig,ax = plt.subplots(figsize=(15,15))
cmap = plt.get_cmap("tab20c")

inner_colors = cmap([0,4,8])
middle_colors = cmap([0,2,4,5,8,9])
outer_colors = cmap([0,1,2,3,4,5,6,7,8,9,10,11])

ax.pie(vals.sum(axis=(1,2)), radius=1-size, colors=inner_colors,wedgeprops=dict(width=1-size, edgecolor='w'),labels=['2000_2004','2010-2014','2016-2020'],labeldistance=.3,rotatelabels=True,explode=None,autopct='%1.1f%%',pctdistance=1)
ax.pie(vals.sum(axis=2).flatten(), radius=1, colors=middle_colors,wedgeprops=dict(width=size, edgecolor='w'),labels=['cloud-to-ground','intracloud']*3,labeldistance=.6,rotatelabels=True,explode=None,autopct='%1.1f%%',pctdistance=1)
ax.pie(vals.flatten(), radius=1+size, colors=outer_colors,wedgeprops=dict(width=size, edgecolor='w'),labels=['positive','negative']*6,labeldistance=.75,rotatelabels=True,explode=None,autopct='%1.1f%%',pctdistance=1)

ax.set(aspect="equal", title='')
plt.show()
#fig.savefig('saved_images/statistics_percent.png',dpi=300)

size = 0.5
fig,ax = plt.subplots(figsize=(15,15))
cmap = plt.get_cmap("tab20c")


inner_colors = cmap([0,4,8])
middle_colors = cmap([0,2,4,5,8,9])
outer_colors = cmap([0,1,2,3,4,5,6,7,8,9,10,11])

ax.pie(vals.sum(axis=(1,2)), radius=1-size, colors=inner_colors,wedgeprops=dict(width=1-size, edgecolor='w'),labels=['2000_2004','2010-2014','2016-2020'],labeldistance=.3,rotatelabels=True,explode=None,autopct=lambda x: '{:.0f}'.format(x*vals.sum()/100),pctdistance=1)
ax.pie(vals.sum(axis=2).flatten(), radius=1, colors=middle_colors,wedgeprops=dict(width=size, edgecolor='w'),labels=['cloud-to-ground','intracloud']*3,labeldistance=.6,rotatelabels=True,explode=None,autopct=lambda x: '{:.0f}'.format(x*vals.sum()/100),pctdistance=1)
ax.pie(vals.flatten(), radius=1+size, colors=outer_colors,wedgeprops=dict(width=size, edgecolor='w'),labels=['positive','negative']*6,labeldistance=.75,rotatelabels=True,explode=None,autopct=lambda x: '{:.0f}'.format(x*vals.sum()/100),pctdistance=1)

ax.set(aspect="equal", title='')
plt.show()
#fig.savefig('saved_images/statistics_value.png',dpi=300)


# %% figure generator for candidacy report
#-------------------------------------------------------------------------------
JSD = [[0,0,0],[0,0,0],[0,0,0]]
WD = [[0,0,0],[0,0,0],[0,0,0]]

years = ['2000 \u2013 2004','2010 \u2013 2014','2016 \u2013 2020']
polarity = ['all','positive','negative']
for i,df in zip([0,1,2],[df1,df2,df3]):
    df = df[df['icloud']=='f']
    #df = df[df['amp_abs']>10]
    for j,col in zip([0,1,2],['amp_abs','amp_pos','amp_neg']):
        jsd, emd, inf, sup, num = distance(list(df[col].groupby(pd.Grouper(freq='1D')).count()))
        JSD[i][j] = jsd
        WD[i][j] = emd



fig, [ax1,ax2] = plt.subplots(1,2,figsize = (20,8))

lns = []
for i,c in zip([0,1,2],['r-o','b-o','r-o']):
    if i == 0:
        continue
    ln, = ax1.plot([0,1,2],[JSD[temp][i] for temp in [0,1,2]],c,label=polarity[i]+f' {source}{"s" if source=="stroke" else "es"}')
    lns.append(ln)
    ax1.set_xticks([0,1,2])
    ax1.set_xticklabels(years)
    ax1.set_xlim([-0.25,2.25])
    #ax1.set_title(f'cloud-to-ground {source}{"s" if source=="stroke" else "es"} per day')
ax1.legend(handles=lns, loc='best')
ax1.set_xlabel("time window")
ax1.set_ylabel("Jensen\u2013Shannon Distance")
ax1.tick_params(axis='y',tickdir='inout',rotation=90)

lns = []
for i,c in zip([0,1,2],['r-o','b-o','r-o']):
    if i == 0:
        continue
    ln, = ax2.plot([0,1,2],[WD[temp][i] for temp in [0,1,2]],c,label=polarity[i]+f' {source}{"s" if source=="stroke" else "es"}')
    lns.append(ln)
    ax2.set_xticks([0,1,2])
    ax2.set_xticklabels(years)
    ax2.set_xlim([-0.25,2.25])
    #ax2.set_title(f'cloud-to-ground {source}{"s" if source=="stroke" else "es"} per day')
ax2.legend(handles=lns, loc='best')
ax2.set_xlabel("time window")
ax2.set_ylabel("Wasserstein Distance")
ax2.tick_params(axis='y',tickdir='inout',rotation=90)
#fig.suptitle(f'cloud-to-ground {source}{"s" if source=="stroke" else "es"} per day')
fig.tight_layout()
#fig.savefig(f'saved_images_new_range/grid_compare_JSD_WD.png',dpi=300,bbox_inches='tight')



# %% figure generator for candidacy report
#-------------------------------------------------------------------------------
JSD = [[0,0,0],[0,0,0],[0,0,0]]
WD = [[0,0,0],[0,0,0],[0,0,0]]

years = ['2000 \u2013 2004','2010 \u2013 2014','2016 \u2013 2020']
polarity = ['all','positive','negative']
for i,df in zip([0,1,2],[df1,df2,df3]):
    df = df[df['icloud']=='f']
    df = df[df['amp_abs']>10]
    for j,col in zip([0,1],['amp_abs','amp_pos']):
        jsd, emd, inf, sup, num = distance(list(df[col].groupby(pd.Grouper(freq='1D')).count()))
        JSD[i][j] = jsd
        WD[i][j] = emd

for i,df in zip([0,1,2],[df1,df2,df3]):
    df = df[df['icloud']=='f']
    #df = df[df['amp_abs']>10]
    for j,col in zip([2],['amp_neg']):
        jsd, emd, inf, sup, num = distance(list(df[col].groupby(pd.Grouper(freq='1D')).count()))
        JSD[i][j] = jsd
        WD[i][j] = emd


fig, [ax1,ax2] = plt.subplots(1,2,figsize = (20,8))

lns = []
for i,c in zip([0,1,2],['r-o','b-o','r-o']):
    if i == 0:
        continue
    ln, = ax1.plot([0,1,2],[JSD[temp][i] for temp in [0,1,2]],c,label=polarity[i]+f' {source}{"s" if source=="stroke" else "es"}')
    lns.append(ln)
    ax1.set_xticks([0,1,2])
    ax1.set_xticklabels(years)
    ax1.set_xlim([-0.25,2.25])
    #ax1.set_title(f'cloud-to-ground {source}{"s" if source=="stroke" else "es"} per day')
ax1.legend(handles=lns, loc='best')
ax1.set_xlabel("time window")
ax1.set_ylabel("Jensen\u2013Shannon Distance")
ax1.tick_params(axis='y',tickdir='inout',rotation=90)

lns = []
for i,c in zip([0,1,2],['r-o','b-o','r-o']):
    if i == 0:
        continue
    ln, = ax2.plot([0,1,2],[WD[temp][i] for temp in [0,1,2]],c,label=polarity[i]+f' {source}{"s" if source=="stroke" else "es"}')
    lns.append(ln)
    ax2.set_xticks([0,1,2])
    ax2.set_xticklabels(years)
    ax2.set_xlim([-0.25,2.25])
    #ax2.set_title(f'cloud-to-ground {source}{"s" if source=="stroke" else "es"} per day')
ax2.legend(handles=lns, loc='best')
ax2.set_xlabel("time window")
ax2.set_ylabel("Wasserstein Distance")
ax2.tick_params(axis='y',tickdir='inout',rotation=90)
#fig.suptitle(f'cloud-to-ground {source}{"s" if source=="stroke" else "es"} over 10 KA per day')
fig.tight_layout()
#fig.savefig(f'saved_images_new_range/grid_compare_JSD_WD_over10k.png',dpi=300,bbox_inches='tight')
# %%
#-------------------------------------------------------------------------------
for polarity,p_text,p_save,limit in zip(['abs','pos','neg'],['','positive ','negative '],['all','pos','neg'],[10000,1000,10000]):
    f,a = plt.subplots(1,1,figsize=(25,14))
    plt.title(f'number of {p_text}cloud-to-ground {source}{"s" if source=="stroke" else "es"}')
    a.plot([0,1],[0,1],'r-*',alpha=0)
    plt.xticks([0.1,0.37,0.64,0.91],['original','thresholded','multiplied_lower','multiplied_higher'])
    plt.yticks([0.135,0.505,0.875],['2016-2020','2010-2014','2000_2004'])
    plt.ylabel('time window')
    #plt.xlabel('time slice size (day)')
    for i,(name,df) in enumerate(zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3])):
        df = df[df['icloud']=='f']
        for j,l in enumerate([np.inf,limit]):
            left = 0.145 + j * 0.19
            bottom = 0.653 - i * 0.25
            width = 0.17
            height = 0.18
            right = left + width
            top = bottom + height
            center_lr = (left + right)/2
            center_tb = (top+bottom)/2
            #a.plot([left,left,right,right],[top,bottom,top,bottom],'ro')
            #a.plot([center_lr],[center_tb],"b*",transform=a.transAxes)
            ax = plt.axes([left,bottom,width,height])
            temp = list(df[f'amp_{polarity}'].groupby(pd.Grouper(freq=f'1D')).count())
            to_process = [x for x in temp if x<l and x>0]

            hist, bins = np.histogram(to_process,bins=50)
            logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
            ax.hist(to_process, bins=logbins)
            ax.set_xscale('log')

            jsd, _, _, _, _ = distance(to_process)

            ax.set_title(ax.get_title() + f'JSD:{jsd:.3f}\nThreshold:{l},\n rho:1, N:{len([x for x in to_process if x<l and x>0])}',fontsize=13)  # df[df[f'amp_{polarity}']>l]


        for j,func in zip([2,3],[np.floor,np.ceil]):
            left = 0.145 + j * 0.19
            bottom = 0.653 - i * 0.25
            width = 0.17
            height = 0.18
            right = left + width
            top = bottom + height
            center_lr = (left + right)/2
            center_tb = (top+bottom)/2
            #a.plot([left,left,right,right],[top,bottom,top,bottom],'ro')
            #a.plot([center_lr],[center_tb],"b*",transform=a.transAxes)
            ax = plt.axes([left,bottom,width,height])
            temp = list(df[f'amp_{polarity}'].groupby(pd.Grouper(freq=f'1D')).count())
            to_process = [x for x in temp if x>0]
            m = max(to_process)
            l = np.power(10,func(np.log10(m)))
            rho = l/m
            to_process = [int(x*rho) for x in to_process if x*rho>=1]

            hist, bins = np.histogram(to_process,bins=50)
            logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
            ax.hist(to_process, bins=logbins)
            ax.set_xscale('log')

            jsd, _, _, _, _ = distance(to_process)

            ax.set_title(ax.get_title() + f'JSD:{jsd:.3f}\nThreshold:{np.inf},\n rho:{rho:.3f}, N:{len([x for x in to_process if x<l and x>0])}',fontsize=13)  # df[df[f'amp_{polarity}']>l]


    #f.savefig(f'saved_images_new_range/grid_histogram_{p_save}.png',dpi=300,bbox_inches='tight')
# %% this one
#-------------------------------------------------------------------------------
#x = np.logspace(1,3,500).astype('uint')
x = np.logspace(0,3,500)
y_t = []
y_r = []
sups_t = []
infs_t = []
nums_t = []
for i,df in enumerate([df1,df2,df3]):
    y_t.append([])
    y_r.append([])
    sups_t.append([])
    infs_t.append([])
    nums_t.append([])
    df = df[df['icloud']=='f']
    for j,col_name in enumerate(['amp_abs','amp_pos','amp_neg']):
        y_t[i].append([])
        y_r[i].append([])
        sups_t[i].append([])
        infs_t[i].append([])
        nums_t[i].append([])
        for freq in x:
            freq = freq.astype('uint')
            jsd, emd, inf, sup, num = distance(list(df[col_name].groupby(pd.Grouper(freq=f'{freq}H')).count()))
            y_t[i][j].append(jsd)
            y_r[i][j].append(emd)
            sups_t[i][j].append(sup)
            infs_t[i][j].append(inf)
            nums_t[i][j].append(num)
# %% this one
#-------------------------------------------------------------------------------
year = ['2000-2004','2010-2014','2016-2020']
polarity = ['all','positive','negative']
i = 0
j = 0
y = y_t[i][j]
z = y_r[i][j]
sups = sups_t[i][j]
infs = infs_t[i][j]
nums = nums_t[i][j]

fig,ax = plt.subplots(figsize = (20,10))


part1 = ax.twinx()
part2 = ax.twinx()
part3 = ax.twinx()

p0, = ax.plot(x,nums,'black',label='number of time slices')
ax.set_ylabel('number of time slices')
ax.set_xlabel('time slice (hours)')

p1, = part1.plot(x,y,':C0',label='JSD',alpha=0.5)

part1.set_ylabel('Jensen Shannon Distance')

p2, = part2.plot(x,sups,'r',label='maximum',alpha=1)
part2.set_ylabel('maximum')

p3, = part3.plot(x,z,'g',label='WD',alpha=1)
part3.set_ylabel('WD')

#plt.xticks(x)
plt.xscale('log')

#lns = [p0,p1,p2,p3]



# right, left, top, bottom
part2.spines['right'].set_position(('outward', 90))
part2.spines['right'].set_visible(True)
part2.yaxis.set_label_position('right')
part2.yaxis.set_ticks_position('right')
part3.spines['right'].set_position(('outward',180))
part3.spines['right'].set_visible(True)
part3.yaxis.set_label_position('right')
part3.yaxis.set_ticks_position('right')



ax.yaxis.label.set_color(p0.get_color())
part1.yaxis.label.set_color(p1.get_color())
part2.yaxis.label.set_color(p2.get_color())
part3.yaxis.label.set_color(p3.get_color())
#fig.tight_layout()

#plt.title(f"{polarity[j]} cloud to ground strokes in {year[i]}")


#part1.axvline(x=72, ymin=0.045, ymax=0.32, color='g', label='axvline - % of full height')

ax.axvline(x=24,ymin=0,ymax=13000)#24hour,1day

plt.xscale('log')

#fig.savefig(f'out_{i}{j}.png',dpi=10*fig.dpi,bbox_inches='tight',pad_inches=1)


# 300 represents number of points to make between T.min and T.max
#xnew = np.logspace(np.log10(x.min()), np.log10(x.max()), 1000)

#spl = make_interp_spline(np.log10(x), y, k=5)  # type: BSpline
#y_smooth = spl(np.log10(xnew))



kernel_size = 30
kernel = np.ones(kernel_size) / kernel_size
#y_smooth = np.convolve(y, kernel, mode='same')
y_smooth = np.convolve(np.pad(y,(kernel_size//2,kernel_size-kernel_size//2-1),mode='edge'), kernel, mode='valid')

z_smooth = np.convolve(np.pad(z,(kernel_size//2,kernel_size-kernel_size//2-1),mode='edge'), kernel, mode='valid')

p4, = part1.plot(x,y_smooth,'C0',label='JSD-smooth',alpha=1)
p5, = part3.plot(x,z_smooth,'g',label='WD-smooth',alpha=1)

p6 = part1.add_patch(mpatches.Rectangle(xy=[24, 0.05], width=96, height=0.04,facecolor='None',edgecolor='green',linewidth=3,alpha=0.9))
#fig.savefig(f'saved_images/out_{i}{j}.png',dpi=10*fig.dpi,bbox_inches='tight',pad_inches=1)
p1.set_alpha(0.5)
p2.set_alpha(1)
p3.set_alpha(0.5)
p4.set_alpha(1)
p5.set_alpha(1)
lns = [p0,p1,p4,p3,p5,p2]
ax.legend(handles=lns, loc='upper center')
#fig.savefig(f'saved_images_new_range/step_5.png',dpi=300,bbox_inches='tight')





# %% figure generator for candidacy report
#-------------------------------------------------------------------------------
year = ['2000-2004','2010-2014','2016-2020']
polarity = ['positive','negative']


f,a = plt.subplots(1,1,figsize=(25,14))

a.plot([0,1],[0,1],'r-*',alpha=0)
plt.xticks([0.20,0.75],[f'positive {source}{"s" if source=="stroke" else "es"}',f'negative {source}{"s" if source=="stroke" else "es"}'])
plt.yticks([0.21,0.56,0.92],['2016-2020','2010-2014','2000_2004'],rotation=90)
#plt.ylabel('time window')
#plt.xlabel('lightning type')
a.tick_params(axis=u'both', which=u'both',length=0)


for i in range(3):
    for j in range(1,3):

        y = y_t[i][j]
        z = y_r[i][j]
        sups = sups_t[i][j]
        infs = infs_t[i][j]
        nums = nums_t[i][j]

        left = 0.145 + (j-1) * 0.385
        bottom = 0.64 - i * 0.24
        width = 0.31
        height = 0.22
        right = left + width
        top = bottom + height
        center_lr = (left + right)/2
        center_tb = (top+bottom)/2

        ax = plt.axes([left,bottom,width,height])


        part1 = ax.twinx()
        part2 = ax.twinx()
        part3 = ax.twinx()

        p0, = ax.plot(x,np.log(nums),'black',label='# of t.s.')
        ax.set_ylabel('number of time slices(log)',size=10)
        ax.set_xlabel('time slice (hour)',size=10)

        p1, = part1.plot(x,y,':C0',label='JSD',alpha=0.5)

        part1.set_ylabel('Jensen Shannon Distance',size=10)

        p2, = part2.plot(x,np.log(sups),'r',label='max',alpha=1)
        part2.set_ylabel('maximum(log)',size=10)

        p3, = part3.plot(x,z,':g',label='WD',alpha=1)
        part3.set_ylabel('Wasserstein Distance',size=10)





        ax.yaxis.set_visible(True)
        ax.tick_params(axis='y',labelsize=10,tickdir='inout',rotation=90)
        #ax.set_yticks([0,4000,8000])

        ax.xaxis.set_visible(True)
        ax.tick_params(axis='x',labelsize=10,tickdir='inout')

        part1.spines['right'].set_visible(False)
        part1.yaxis.set_visible(True)
        part1.tick_params(axis='y',labelsize=10,tickdir='inout',rotation=90)
        part1.set_yticks([0,0.05,0.1,0.15,0.2,0.25])


        part2.spines['right'].set_position(('outward', 70))
        part2.spines['right'].set_visible(True)
        part2.yaxis.set_visible(True)
        part2.yaxis.set_label_position('right')
        part2.yaxis.set_ticks_position('right')
        part2.tick_params(axis='y',labelsize=10,tickdir='inout',rotation=90)
        #part2.set_yticks([0,10000,100000])

        part3.spines['right'].set_position(('outward',35))
        part3.spines['right'].set_visible(True)
        part3.yaxis.set_visible(True)
        part3.yaxis.set_label_position('right')
        part3.yaxis.set_ticks_position('right')
        part3.tick_params(axis='y',labelsize=10,tickdir='inout',rotation=90)



        ax.yaxis.label.set_color(p0.get_color())
        part1.yaxis.label.set_color(p1.get_color())
        part2.yaxis.label.set_color(p2.get_color())
        part3.yaxis.label.set_color(p3.get_color())




        ax.set_xscale('log')
        ax.set_xlim([0.9,1100])



        kernel_size = 30
        kernel = np.ones(kernel_size) / kernel_size

        y_smooth = np.convolve(np.pad(y,(kernel_size//2,kernel_size-kernel_size//2-1),mode='edge'), kernel, mode='valid')
        z_smooth = np.convolve(np.pad(z,(kernel_size//2,kernel_size-kernel_size//2-1),mode='edge'), kernel, mode='valid')

        p4, = part1.plot(x,y_smooth,'C0',label='JSD',alpha=1)
        p5, = part3.plot(x,z_smooth,'g',label='WD',alpha=1)


        p1.set_alpha(0.5)
        p2.set_alpha(1)
        p3.set_alpha(0.5)
        p4.set_alpha(1)
        p5.set_alpha(1)
        lns = [p0,p4,p5,p2]
        ax.annotate('1 Day', xy=(24,3.7), xytext=(18.5,np.max(np.log(nums))),arrowprops={'arrowstyle': '->', 'lw': 4, 'color': 'blue'},va='center')
        ax.legend(handles=lns, loc='lower right',prop={"size":10})
#f.savefig(f'saved_images_new_range/grid_sweep_JSD_WD.png',dpi=300,bbox_inches='tight')

# %%
#-------------------------------------------------------------------------------
year = ['2000-2004','2010-2014','2016-2020']
polarity = ['positive','negative']


f,a = plt.subplots(1,1,figsize=(25,14))

a.plot([0,1],[0,1],'r-*',alpha=0)
plt.xticks([0.21,0.75],[f'positive {source}{"s" if source=="stroke" else "es"}',f'negative {source}{"s" if source=="stroke" else "es"}'])
plt.yticks([0.13,0.50,0.87],['2016-2020','2010-2014','2000_2004'])
plt.ylabel('time window')
plt.xlabel('lightning type')


for i in range(3):
    for j in range(1,3):

        y = y_t[i][j]
        z = y_r[i][j]
        sups = sups_t[i][j]
        infs = infs_t[i][j]
        nums = nums_t[i][j]

        left = 0.14 + (j-1) * 0.382
        bottom = 0.64 - i * 0.25
        width = 0.36
        height = 0.22
        right = left + width
        top = bottom + height
        center_lr = (left + right)/2
        center_tb = (top+bottom)/2

        ax = plt.axes([left,bottom,width,height])


        part1 = ax.twinx()
        part2 = ax.twinx()
        part3 = ax.twinx()

        p0, = ax.plot(x,nums,'black',label='number of time slices')
        ax.set_ylabel('number of time slices')
        ax.set_xlabel('time slice (hour)')

        p1, = part1.plot(x,y,':C0',label='JSD',alpha=0.5)

        part1.set_ylabel('Jensen Shannon Distance')

        p2, = part2.plot(x,sups,'r',label='maximum',alpha=1)
        part2.set_ylabel('maximum')

        p3, = part3.plot(x,z,':g',label='Wasserstein Distance',alpha=1)
        part3.set_ylabel('WD')

        plt.xscale('log')



        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)

        part1.spines['right'].set_visible(False)
        part1.yaxis.set_visible(False)

        part2.spines['right'].set_position(('outward', 90))
        part2.spines['right'].set_visible(False)
        part2.yaxis.set_visible(False)
        part2.yaxis.set_label_position('right')
        part2.yaxis.set_ticks_position('right')

        part3.spines['right'].set_position(('outward',180))
        part3.spines['right'].set_visible(False)
        part3.yaxis.set_visible(False)
        part3.yaxis.set_label_position('right')
        part3.yaxis.set_ticks_position('right')



        ax.yaxis.label.set_color(p0.get_color())
        part1.yaxis.label.set_color(p1.get_color())
        part2.yaxis.label.set_color(p2.get_color())
        part3.yaxis.label.set_color(p3.get_color())
        #fig.tight_layout()

        #plt.title(f"{polarity[j]} cloud to ground strokes in {year[i]}")


        #part1.axvline(x=72, ymin=0.045, ymax=0.32, color='g', label='axvline - % of full height')



        plt.xscale('log')

        #fig.savefig(f'out_{i}{j}.png',dpi=10*fig.dpi,bbox_inches='tight',pad_inches=1)


        # 300 represents number of points to make between T.min and T.max
        #xnew = np.logspace(np.log10(x.min()), np.log10(x.max()), 1000)

        #spl = make_interp_spline(np.log10(x), y, k=5)  # type: BSpline
        #y_smooth = spl(np.log10(xnew))



        kernel_size = 30
        kernel = np.ones(kernel_size) / kernel_size
        #y_smooth = np.convolve(y, kernel, mode='same')
        y_smooth = np.convolve(np.pad(y,(kernel_size//2,kernel_size-kernel_size//2-1),mode='edge'), kernel, mode='valid')
        z_smooth = np.convolve(np.pad(z,(kernel_size//2,kernel_size-kernel_size//2-1),mode='edge'), kernel, mode='valid')

        p4, = part1.plot(x,y_smooth,'C0',label='JSD-smooth',alpha=1)
        p5, = part3.plot(x,z_smooth,'g',label='JSD-smooth',alpha=1)

        #p6 = part1.add_patch(mpatches.Rectangle(xy=[24, 0.05], width=96, height=0.04,facecolor='None',edgecolor='green',linewidth=3,alpha=0.9))
        #fig.savefig(f'saved_images/out_{i}{j}.png',dpi=10*fig.dpi,bbox_inches='tight',pad_inches=1)
        p1.set_alpha(0.5)
        p2.set_alpha(1)
        p3.set_alpha(0.5)
        p4.set_alpha(1)
        p5.set_alpha(1)
        lns = [p0,p1,p4,p3,p5,p2]
        ax.annotate('1 Day', xy=(24,0), xytext=(18.5,np.max(nums)),arrowprops={'arrowstyle': '->', 'lw': 4, 'color': 'blue'},va='center')
        #ax.legend(handles=lns, loc='lower right')
#f.savefig(f'saved_images_new_range/grid_out_window.png',dpi=300,bbox_inches='tight')


# %% figure generator for candidacy report and paper
#-------------------------------------------------------------------------------
from importlib import reload
import benford
reload(benford)
year = ['2000-2004','2010-2014','2016-2020']
polarity = ['negative','positive']


f,a = plt.subplots(1,1,figsize=(25,14))
plt.title(f'{source}{"s" if source=="stroke" else "es"} per day')
a.plot([0,1],[0,1],'r-*',alpha=0)
plt.xticks([0.21,0.75],[f'negative {source}{"s" if source=="stroke" else "es"}',f'positive {source}{"s" if source=="stroke" else "es"}'],size=20)
plt.yticks([0.13,0.50,0.87],['2016 \u2013 2020','2010 \u2013 2014','2000 \u2013 2004'],rotation=90,ha='right',va='center',size=20)
plt.ylabel('time window',size=30)
plt.xlabel('lightning type',size=30)

container = [df1,df2,df3]
polarity_name = ['neg','pos']

for i in range(3):
    df = container[i]
    df = df[df['icloud']=='f']
    for j in range(1,3):

        #y = y_t[i][j]
        #z = y_r[i][j]
        #sups = sups_t[i][j]
        #infs = infs_t[i][j]
        #nums = nums_t[i][j]

        left = 0.15 + (j-1) * 0.382
        bottom = 0.645 - i * 0.25
        width = 0.35
        height = 0.21
        right = left + width
        top = bottom + height
        center_lr = (left + right)/2
        center_tb = (top+bottom)/2

        ax = plt.axes([left,bottom,width,height])
        benford.benford(list(df[f'amp_{polarity_name[j-1]}'].groupby(pd.Grouper(freq=f'1D')).count()),f'',ax=ax)
        ax.axes.set_xticks(range(1,10))
        ax.set_title(ax.get_title().lstrip().replace('\n',', ') )
        ax.legend(prop={'size':20})
        ax.set_ylim([0,.4])

#f.savefig(f'saved_images_new_range/grid_results.png',dpi=300,bbox_inches='tight')
#f.savefig(f'saved_images_new_range/grid_results_scaled.png',dpi=300,bbox_inches='tight')


# %%
#-------------------------------------------------------------------------------
for polarity,p_text,p_save,limit in zip(['abs','pos','neg'],['','positive ','negative '],['all','pos','neg'],[10000,1000,10000]):
    f,a = plt.subplots(1,1,figsize=(25,14))
    plt.title(f'number of {p_text}cloud-to-ground {source}{"s" if source=="stroke" else "es"}')
    a.plot([0,1],[0,1],'r-*',alpha=0)
    plt.xticks([0.21,0.75],['original','thresholded'])
    plt.yticks([0.11,0.48,0.85],['2016-2020','2010-2014','2000_2004'])
    plt.ylabel('time window')
    #plt.xlabel('time slice size (day)')
    for i,(name,df) in enumerate(zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3])):
        df = df[df['icloud']=='f']
        for j,l in enumerate([np.inf,limit]):
            left = 0.15 + j * 0.38
            bottom = 0.65 - i * 0.25
            width = 0.32
            height = 0.18
            right = left + width
            top = bottom + height
            center_lr = (left + right)/2
            center_tb = (top+bottom)/2
            #a.plot([left,left,right,right],[top,bottom,top,bottom],'ro')
            #a.plot([center_lr],[center_tb],"b*",transform=a.transAxes)
            ax = plt.axes([left,bottom,width,height])
            to_process = list(df[f'amp_{polarity}'].groupby(pd.Grouper(freq=f'1D')).count())
            benford(to_process,f'',limit=l,ax=ax)

            ax.set_title(ax.get_title() + f', Threshold:{l}, N:{len([x for x in to_process if x<l and x>0])}')  # df[df[f'amp_{polarity}']>l]
            ax.axes.set_xticks(range(1,10))

    #f.savefig(f'saved_images_new_range/grid_benford_limit_{p_save}.png',dpi=300,bbox_inches='tight')
# %%
#-------------------------------------------------------------------------------
for polarity,p_text,p_save,limit in zip(['abs','pos','neg'],['','positive ','negative '],['all','pos','neg'],[10000,1000,10000]):
    f,a = plt.subplots(1,1,figsize=(25,14))
    plt.title(f'number of {p_text}cloud-to-ground {source}{"s" if source=="stroke" else "es"}')
    a.plot([0,1],[0,1],'r-*',alpha=0)
    plt.xticks([0.12,0.48,0.85],['original','thresholded','multiplied'])
    plt.yticks([0.135,0.505,0.875],['2016-2020','2010-2014','2000_2004'])
    plt.ylabel('time window')
    #plt.xlabel('time slice size (day)')
    for i,(name,df) in enumerate(zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3])):
        df = df[df['icloud']=='f']
        for j,l in enumerate([np.inf,limit]):
            left = 0.165 + j * 0.255
            bottom = 0.653 - i * 0.25
            width = 0.18
            height = 0.18
            right = left + width
            top = bottom + height
            center_lr = (left + right)/2
            center_tb = (top+bottom)/2
            #a.plot([left,left,right,right],[top,bottom,top,bottom],'ro')
            #a.plot([center_lr],[center_tb],"b*",transform=a.transAxes)
            ax = plt.axes([left,bottom,width,height])
            to_process = list(df[f'amp_{polarity}'].groupby(pd.Grouper(freq=f'1D')).count())
            benford(to_process,f'',limit=l,ax=ax)

            ax.set_title(ax.get_title() + f', Threshold:{l},\n rho:1, N:{len([x for x in to_process if x<l and x>0])}',fontsize=13)  # df[df[f'amp_{polarity}']>l]
            ax.axes.set_xticks(range(1,10))

        j=2
        left = 0.165 + j * 0.255
        bottom = 0.653 - i * 0.25
        width = 0.18
        height = 0.18
        right = left + width
        top = bottom + height
        center_lr = (left + right)/2
        center_tb = (top+bottom)/2
        #a.plot([left,left,right,right],[top,bottom,top,bottom],'ro')
        #a.plot([center_lr],[center_tb],"b*",transform=a.transAxes)
        ax = plt.axes([left,bottom,width,height])
        to_process = list(df[f'amp_{polarity}'].groupby(pd.Grouper(freq=f'1D')).count())
        rho = limit/max(to_process)
        benford(to_process,f'',rho=rho,ax=ax)

        ax.set_title(ax.get_title() + f', Threshold:{np.inf},\n rho:{rho:.3f}, N:{len([x for x in to_process if x<l and x>0])}',fontsize=13)  # df[df[f'amp_{polarity}']>l]
        ax.axes.set_xticks(range(1,10))

    #f.savefig(f'saved_images_new_range/grid_benford_limit_multiplier_{p_save}.png',dpi=300,bbox_inches='tight')

# %%
#-------------------------------------------------------------------------------
for polarity,p_text,p_save,limit in zip(['abs','pos','neg'],['','positive ','negative '],['all','pos','neg'],[10000,1000,10000]):
    f,a = plt.subplots(1,1,figsize=(25,14))
    plt.title(f'number of {p_text}cloud-to-ground {source}{"s" if source=="stroke" else "es"}')
    a.plot([0,1],[0,1],'r-*',alpha=0)
    plt.xticks([0.1,0.37,0.64,0.91],['original','thresholded','multiplied_lower','multiplied_higher'])
    plt.yticks([0.135,0.505,0.875],['2016-2020','2010-2014','2000_2004'])
    plt.ylabel('time window')
    #plt.xlabel('time slice size (day)')
    for i,(name,df) in enumerate(zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3])):
        df = df[df['icloud']=='f']
        df = df[df['amp_abs']>10]
        for j,l in enumerate([np.inf,limit]):
            left = 0.145 + j * 0.19
            bottom = 0.653 - i * 0.25
            width = 0.17
            height = 0.18
            right = left + width
            top = bottom + height
            center_lr = (left + right)/2
            center_tb = (top+bottom)/2
            #a.plot([left,left,right,right],[top,bottom,top,bottom],'ro')
            #a.plot([center_lr],[center_tb],"b*",transform=a.transAxes)
            ax = plt.axes([left,bottom,width,height])
            to_process = list(df[f'amp_{polarity}'].groupby(pd.Grouper(freq=f'1D')).count())
            benford(to_process,f'',limit=l,ax=ax)

            ax.set_title(ax.get_title() + f', Threshold:{l},\n rho:1, N:{len([x for x in to_process if x<l and x>0])}',fontsize=13)  # df[df[f'amp_{polarity}']>l]
            ax.axes.set_xticks(range(1,10))

        for j,func in zip([2,3],[np.floor,np.ceil]):
            left = 0.145 + j * 0.19
            bottom = 0.653 - i * 0.25
            width = 0.17
            height = 0.18
            right = left + width
            top = bottom + height
            center_lr = (left + right)/2
            center_tb = (top+bottom)/2
            #a.plot([left,left,right,right],[top,bottom,top,bottom],'ro')
            #a.plot([center_lr],[center_tb],"b*",transform=a.transAxes)
            ax = plt.axes([left,bottom,width,height])
            to_process = list(df[f'amp_{polarity}'].groupby(pd.Grouper(freq=f'1D')).count())
            m = max(to_process)
            l = np.power(10,func(np.log10(m)))
            rho = l/m
            benford(to_process,f'',rho=rho,ax=ax)

            ax.set_title(ax.get_title() + f', Threshold:{np.inf},\n rho:{rho:.3f}, N:{len([x for x in to_process if x<l and x>0])}',fontsize=13)  # df[df[f'amp_{polarity}']>l]
            ax.axes.set_xticks(range(1,10))

    #f.savefig(f'saved_images_new_range/grid_benford_limit_multiplier_2_times_{p_save}.png',dpi=300,bbox_inches='tight')
# %%
f,a = plt.subplots(1,1,figsize=(25,14))
plt.title(f'Thresholding effect')
a.plot([0,1],[0,1],'r-*',alpha=0)
plt.xticks([0.12,0.48,0.85],[f'all {source}{"s" if source=="stroke" else "es"}',f'positive {source}{"s" if source=="stroke" else "es"}',f'negative {source}{"s" if source=="stroke" else "es"}'])
plt.yticks([0.135,0.505,0.875],['2016-2020','2010-2014','2000_2004'])
plt.ylabel('time window')
plt.xlabel('lightning type')
for j,(polarity,p_text,p_save,sweep) in enumerate(zip(['abs','pos','neg'],['','positive ','negative '],['all','pos','neg'],[(300,30000),(100,10000),(300,30000)])):
    for i,(name,df) in enumerate(zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3])):
        df = df[df['icloud']=='f']
        left = 0.165 + j * 0.255
        bottom = 0.68 - i * 0.25
        width = 0.18
        height = 0.18
        right = left + width
        top = bottom + height
        center_lr = (left + right)/2
        center_tb = (top+bottom)/2
        #a.plot([left,left,right,right],[top,bottom,top,bottom],'ro')
        #a.plot([center_lr],[center_tb],"b*",transform=a.transAxes)
        ax = plt.axes([left,bottom,width,height])
        #benford(list(df[f'amp_{polarity}'].groupby(pd.Grouper(freq=f'{freq}D')).count()),f'',limit=limit,ax=ax)
        numbers = list(df[f'amp_{polarity}'].groupby(pd.Grouper(freq=f'1D')).count())
        M = max(numbers)
        m = min(numbers)
        x = np.logspace(np.log10(sweep[0]),np.log10(sweep[1]),100)
        y = np.zeros_like(x)
        nums = np.zeros_like(x)
        for index,limit in enumerate(x):
            jsd, _, _, _, num = distance(numbers,limit=limit)
            y[index] = jsd
            nums[index] = num
        p0, = ax.plot(x,y,label='JSD')
        ax_ = ax.twinx()
        p1, = ax_.plot(x,nums,'g',label='threshold')
        ax_.set_ylabel('number of time slices')
        p2 = ax.axvline(x=M, ymin=0.0, ymax=1.0, color='r',label='maximum number')
        ax.set_xlabel('enforced upper limit(threshold)')
        ax.set_ylabel('jsd')
        ax.set_xscale('log')
        #ax.axes.set_xticks(range(1,10))
        ax.yaxis.label.set_color(p0.get_color())
        ax_.yaxis.label.set_color(p1.get_color())

    #a.legend(handles=[p0,p1,p2],loc = 'upper left')
    #f.savefig(f'saved_images_new_range/grid_limit.png',dpi=300)





















# %%
#-------------------------------------------------------------------------------
x = np.linspace(0,60,31)

y_t = []
sups_t = []
infs_t = []
nums_t = []
for i,df in enumerate([df1,df2,df3]):
    y_t.append([])
    sups_t.append([])
    infs_t.append([])
    nums_t.append([])
    df = df[df['icloud']=='f']
    for j,col_name in enumerate(['amp_abs','amp_pos','amp_neg']):
        dff = df[np.isnan(df[col_name]) == False ]
        print(min(dff[col_name]),max(dff[col_name]))
        y_t[i].append([])
        sups_t[i].append([])
        infs_t[i].append([])
        nums_t[i].append([])
        for lim in x:
            #freq = freq.astype('uint')
            dff = dff[dff[col_name]>lim]
            jsd, emd, inf, sup, num = distance(list(dff[col_name].groupby(pd.Grouper(freq=f'1D')).count()))
            y_t[i][j].append(jsd)
            sups_t[i][j].append(sup)
            infs_t[i][j].append(inf)
            nums_t[i][j].append(num)
# %%
#-------------------------------------------------------------------------------
year = ['2000-2004','2010-2014','2016-2020']
polarity = ['all','positive','negative']
i = 0
j = 1
y = y_t[i][j]
sups = sups_t[i][j]
infs = infs_t[i][j]
nums = nums_t[i][j]

fig,ax = plt.subplots(figsize = (20,10))


part1 = ax.twinx()
part2 = ax.twinx()
#part3 = ax.twinx()

p0, = ax.plot(x,nums,'black',label='number of time slices')
ax.set_ylabel('number of time slices')
ax.set_xlabel('current\'s lower limit')

p1, = part1.plot(x,y,'C0',label='JSD',alpha=0.5)

part1.set_ylabel('Jensen Shannon Distance')

p2, = part2.plot(x,sups,'r',label='maximum',alpha=1)
part2.set_ylabel('maximum')

#p3, = part3.plot(x,infs,'g',label='minimum',alpha=1)
#part3.set_ylabel('minimum')

#plt.xticks(x)
#plt.xscale('log')

#lns = [p0,p1,p2,p3]



# right, left, top, bottom
part2.spines['right'].set_position(('outward', 90))
part2.spines['right'].set_visible(True)
part2.yaxis.set_label_position('right')
part2.yaxis.set_ticks_position('right')
#part3.spines['right'].set_position(('outward',180))
#part3.spines['right'].set_visible(True)
#part3.yaxis.set_label_position('right')
#part3.yaxis.set_ticks_position('right')



ax.yaxis.label.set_color(p0.get_color())
part1.yaxis.label.set_color(p1.get_color())
part2.yaxis.label.set_color(p2.get_color())
#part3.yaxis.label.set_color(p3.get_color())
#fig.tight_layout()

#plt.title(f"{polarity[j]} cloud to ground strokes in {year[i]}")


#part1.axvline(x=72, ymin=0.045, ymax=0.32, color='g', label='axvline - % of full height')



#plt.xscale('log')

#fig.savefig(f'out_{i}{j}.png',dpi=10*fig.dpi,bbox_inches='tight',pad_inches=1)


# 300 represents number of points to make between T.min and T.max
#xnew = np.logspace(np.log10(x.min()), np.log10(x.max()), 1000)

#spl = make_interp_spline(np.log10(x), y, k=5)  # type: BSpline
#y_smooth = spl(np.log10(xnew))



kernel_size = 5
kernel = np.ones(kernel_size) / kernel_size
y_smooth = np.convolve(y, kernel, mode='same')
y_smooth = np.convolve(np.pad(y,(kernel_size//2,kernel_size-kernel_size//2-1),mode='edge'), kernel, mode='valid')

p3, = part1.plot(x,y_smooth,'C0',label='JSD-smooth',alpha=1)


#fig.savefig(f'saved_images/out_{i}{j}.png',dpi=10*fig.dpi,bbox_inches='tight',pad_inches=1)
p1.set_alpha(0.5)
p2.set_alpha(1)
p3.set_alpha(1)

part2.set_yscale('log')
part2.hlines(1000,0,60,'r',alpha=0.6,linestyle='dashed')
part2.hlines(10000,0,60,'r',alpha=0.6,linestyle='dashed')
part1.vlines(10,0.035,0.095,'C0',alpha=0.6,linestyle='dashed')
part1.vlines(42,0.035,0.095,'C0',alpha=0.6,linestyle='dashed')

lns = [p0,p1,p3,p2]
ax.legend(handles=lns, loc='upper right')
#fig.savefig(f'saved_images_new_range/step_5.png',dpi=300,bbox_inches='tight')















# %% figure generator for paper atmoshpere
#-------------------------------------------------------------------------------
year = ['2000-2004','2010-2014','2016-2020']
polarity = ['positive','negative']


f,a = plt.subplots(1,1,figsize=(27,6))

#a.plot([0,1],[0,1],'r-*',alpha=0)
#plt.xticks([0.20,0.75],[f'positive {source}{"s" if source=="stroke" else "es"}',f'negative {source}{"s" if source=="stroke" else "es"}'])
#plt.yticks([0.21,0.56,0.92],['2016-2020','2010-2014','2000_2004'],rotation=90)
#plt.ylabel('time window')
#plt.xlabel('lightning type')
a.tick_params(axis=u'both', which=u'both',length=0)
a.axes.set_visible(False)


for i in [2]:
    for j in range(1,3):

        y = y_t[i][j]
        z = y_r[i][j]
        sups = sups_t[i][j]
        infs = infs_t[i][j]
        nums = nums_t[i][j]

        left = 0.145 + (j-1) * 0.39
        bottom = 0.64 - i * 0.24
        width = 0.31
        height = 0.72
        right = left + width
        top = bottom + height
        center_lr = (left + right)/2
        center_tb = (top+bottom)/2

        ax = plt.axes([left,bottom,width,height])
        ax.set_title(polarity[-j+2])


        part1 = ax.twinx()
        part2 = ax.twinx()
        part3 = ax.twinx()

        p0, = ax.plot(x,np.log(nums),'black',label='# of t.s.')
        ax.set_ylabel('number of time slices (log)',size=20)
        ax.set_xlabel('time slice (hours)',size=20)

        p1, = part1.plot(x,y,':C0',label='JSD',alpha=0.5)

        part1.set_ylabel('Jensen Shannon Distance',size=17)

        p2, = part2.plot(x,np.log(sups),'r',label='max',alpha=1)
        part2.set_ylabel('maximum (log)',size=17)

        p3, = part3.plot(x,z,':g',label='WD',alpha=1)
        part3.set_ylabel('Wasserstein Distance',size=17)





        ax.yaxis.set_visible(True)
        ax.tick_params(axis='y',labelsize=10,tickdir='inout',rotation=90)
        #ax.set_yticks([0,4000,8000])

        ax.xaxis.set_visible(True)
        ax.tick_params(axis='x',labelsize=10,tickdir='inout')

        part1.spines['right'].set_visible(False)
        part1.yaxis.set_visible(True)
        part1.tick_params(axis='y',labelsize=10,tickdir='inout',rotation=90)
        part1.set_yticks([0,0.05,0.1,0.15,0.2,0.25])


        part2.spines['right'].set_position(('outward', 70))
        part2.spines['right'].set_visible(True)
        part2.yaxis.set_visible(True)
        part2.yaxis.set_label_position('right')
        part2.yaxis.set_ticks_position('right')
        part2.tick_params(axis='y',labelsize=10,tickdir='inout',rotation=90)
        #part2.set_yticks([0,10000,100000])

        part3.spines['right'].set_position(('outward',35))
        part3.spines['right'].set_visible(True)
        part3.yaxis.set_visible(True)
        part3.yaxis.set_label_position('right')
        part3.yaxis.set_ticks_position('right')
        part3.tick_params(axis='y',labelsize=10,tickdir='inout',rotation=90)



        ax.yaxis.label.set_color(p0.get_color())
        part1.yaxis.label.set_color(p1.get_color())
        part2.yaxis.label.set_color(p2.get_color())
        part3.yaxis.label.set_color(p3.get_color())




        ax.set_xscale('log')
        ax.set_xlim([0.9,1100])



        kernel_size = 30
        kernel = np.ones(kernel_size) / kernel_size

        y_smooth = np.convolve(np.pad(y,(kernel_size//2,kernel_size-kernel_size//2-1),mode='edge'), kernel, mode='valid')
        z_smooth = np.convolve(np.pad(z,(kernel_size//2,kernel_size-kernel_size//2-1),mode='edge'), kernel, mode='valid')

        p4, = part1.plot(x,y_smooth,'C0',label='JSD',alpha=1)
        p5, = part3.plot(x,z_smooth,'g',label='WD',alpha=1)


        p1.set_alpha(0.5)
        p2.set_alpha(1)
        p3.set_alpha(0.5)
        p4.set_alpha(1)
        p5.set_alpha(1)
        lns = [p0,p4,p5,p2]
        ax.annotate('1 Day', xy=(24,3.7), xytext=(18.5,np.max(np.log(nums))),arrowprops={'arrowstyle': '->', 'lw': 4, 'color': 'blue'},va='center')
        ax.legend(handles=lns, loc='lower right',prop={"size":10})
#f.savefig(f'saved_images_new_range/grid_sweep_JSD_WD_reduced_old.png',dpi=300,bbox_inches='tight')
# %% figure generator for paper atmoshpere
#-------------------------------------------------------------------------------
year = ['2000-2004','2010-2014','2016-2020']
polarity = ['positive','negative']


f,a = plt.subplots(1,1,figsize=(27,15))

#a.plot([0,1],[0,1],'r-*',alpha=0)
#plt.xticks([0.20,0.75],[f'positive {source}{"s" if source=="stroke" else "es"}',f'negative {source}{"s" if source=="stroke" else "es"}'])
#plt.yticks([0.21,0.56,0.92],['2016-2020','2010-2014','2000_2004'],rotation=90)
#plt.ylabel('time window')
#plt.xlabel('lightning type')
a.tick_params(axis=u'both', which=u'both',length=0)
a.axes.set_visible(False)


for i in [2]:
    for j in range(1,3):

        y = y_t[i][j]
        z = y_r[i][j]
        sups = sups_t[i][j]
        infs = infs_t[i][j]
        nums = nums_t[i][j]

        left = 0.145 + 0 * 0.39
        bottom = 0.64 - (j-1) * 0.52
        width = 0.71
        height = 0.42
        right = left + width
        top = bottom + height
        center_lr = (left + right)/2
        center_tb = (top+bottom)/2

        ax = plt.axes([left,bottom,width,height])
        ax.set_title(polarity[-j+2] + ' strokes',size=30)


        part1 = ax.twinx()
        part2 = ax.twinx()
        part3 = ax.twinx()

        p0, = ax.plot(x,np.log(nums),'black',label='# of t.s.')
        ax.set_ylabel('number of time slices (log)',size=25)
        ax.set_xlabel('time slice (hours)',size=25)

        p1, = part1.plot(x,y,':C0',label='JSD',alpha=0.5)

        part1.set_ylabel('Jensen Shannon Distance',size=25)

        p2, = part2.plot(x,np.log(sups),'r',label='max',alpha=1)
        part2.set_ylabel('maximum (log)',size=25)

        p3, = part3.plot(x,z,':g',label='WD',alpha=1)
        part3.set_ylabel('Wasserstein Distance',size=25)





        ax.yaxis.set_visible(True)
        ax.tick_params(axis='y',labelsize=20,tickdir='inout',rotation=90)
        #ax.set_yticks([0,4000,8000])

        ax.xaxis.set_visible(True)
        ax.tick_params(axis='x',labelsize=20,tickdir='inout')

        part1.spines['right'].set_visible(False)
        part1.yaxis.set_visible(True)
        part1.tick_params(axis='y',labelsize=20,tickdir='inout',rotation=90)
        part1.set_yticks([0,0.05,0.1,0.15,0.2,0.25])


        part2.spines['right'].set_position(('outward', 110))
        part2.spines['right'].set_visible(True)
        part2.yaxis.set_visible(True)
        part2.yaxis.set_label_position('right')
        part2.yaxis.set_ticks_position('right')
        part2.tick_params(axis='y',labelsize=20,tickdir='inout',rotation=90)
        #part2.set_yticks([0,10000,100000])

        part3.spines['right'].set_position(('outward',55))
        part3.spines['right'].set_visible(True)
        part3.yaxis.set_visible(True)
        part3.yaxis.set_label_position('right')
        part3.yaxis.set_ticks_position('right')
        part3.tick_params(axis='y',labelsize=20,tickdir='inout',rotation=90)



        ax.yaxis.label.set_color(p0.get_color())
        part1.yaxis.label.set_color(p1.get_color())
        part2.yaxis.label.set_color(p2.get_color())
        part3.yaxis.label.set_color(p3.get_color())




        ax.set_xscale('log')
        ax.set_xlim([0.9,1100])



        kernel_size = 30
        kernel = np.ones(kernel_size) / kernel_size

        y_smooth = np.convolve(np.pad(y,(kernel_size//2,kernel_size-kernel_size//2-1),mode='edge'), kernel, mode='valid')
        z_smooth = np.convolve(np.pad(z,(kernel_size//2,kernel_size-kernel_size//2-1),mode='edge'), kernel, mode='valid')

        p4, = part1.plot(x,y_smooth,'C0',label='JSD',alpha=1)
        p5, = part3.plot(x,z_smooth,'g',label='WD',alpha=1)


        p1.set_alpha(0.5)
        p2.set_alpha(1)
        p3.set_alpha(0.5)
        p4.set_alpha(1)
        p5.set_alpha(1)
        lns = [p0,p4,p5,p2]
        ax.annotate('1 Day', xy=(24,3.7), xytext=(21.1,np.max(np.log(nums))),arrowprops={'arrowstyle': '->', 'lw': 4, 'color': 'blue'},va='center')
        ax.legend(handles=lns, loc='lower right',prop={"size":20})
f.savefig(f'saved_images_new_range/grid_sweep_JSD_WD_reduced.png',dpi=300,bbox_inches='tight')


















# %%
#-------------------------------------------------------------------------------
def ploter(i,j,fig):

    ax = fig.add_subplot(111)
    year = ['2000-2004','2010-2014','2016-2020']
    polarity = ['all','positive','negative']
    y = y_t[i][j]
    sups = sups_t[i][j]
    infs = infs_t[i][j]
    nums = nums_t[i][j]




    part1 = ax.twinx()


    p0, = ax.plot(x,y,'C0',label='JSD',alpha=0.5,linestyle='dashed')
    ax.set_ylabel('Jensen Shannon Distance')
    ax.set_xlabel('current\'s lower limit (kA)')
    ax.set_ylim([0.015,0.1])
    #ax.set_xlim([0,60])



    p1, = part1.plot(x,sups,'r',label='maximum',alpha=1)
    part1.set_ylabel('maximum')



    # right, left, top, bottom

    part1.spines['right'].set_visible(True)
    part1.yaxis.set_label_position('right')
    part1.yaxis.set_ticks_position('right')




    ax.yaxis.label.set_color(p0.get_color())
    part1.yaxis.label.set_color(p1.get_color())


    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    y_smooth = np.convolve(y, kernel, mode='same')
    y_smooth = np.convolve(np.pad(y,(kernel_size//2,kernel_size-kernel_size//2-1),mode='edge'), kernel, mode='valid')

    p2, = ax.plot(x,y_smooth,'C0',label='JSD-smooth',alpha=1)
    p1.set_alpha(0.5)

    lns = [p0,p1,p2]
    ax.legend(handles=lns, loc='upper right')




_VARS = {'window': False,
         'fig_agg': False,
         'pltFig': False,
         'i': 0,
         'j': 0}


plt.style.use('Solarize_Light2')

# Helper Functions


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


# \\  -------- PYSIMPLEGUI -------- //

AppFont = 'Any 16'
SliderFont = 'Any 14'
sg.theme('black')

# New layout with slider and padding

layout = [[sg.Canvas(key='figCanvas', background_color='#FDF6E3')],
            [sg.Text(text="   2000-2004",
                     font=SliderFont,
                     background_color='#FDF6E3',
                     pad=((0, 0), (10, 0)),
                     text_color='Black'),
            sg.Text(text="2010-2014\t",
                     font=SliderFont,
                     background_color='#FDF6E3',
                     pad=((0, 0), (10, 0)),
                     text_color='Black'),
            sg.Text(text="2016-2020\t",
                     font=SliderFont,
                     background_color='#FDF6E3',
                     pad=((0, 0), (10, 0)),
                     text_color='Black')],
          [sg.Text(text="year",
                   font=SliderFont,
                   background_color='#FDF6E3',
                   pad=((0, 0), (10, 0)),
                   text_color='Black'),
           sg.Slider(range=(0, 2), orientation='h', size=(34, 20),
                     default_value=0,
                     background_color='#FDF6E3',
                     text_color='Black',
                     key='-Slider1-',
                     enable_events=True,
                     disable_number_display=True)],


            [sg.Text(text="   all\t",
                     font=SliderFont,
                     background_color='#FDF6E3',
                     pad=((0, 0), (10, 0)),
                     text_color='Black'),
            sg.Text(text="positive\t",
                     font=SliderFont,
                     background_color='#FDF6E3',
                     pad=((0, 0), (10, 0)),
                     text_color='Black'),
            sg.Text(text="negative",
                     font=SliderFont,
                     background_color='#FDF6E3',
                     pad=((0, 0), (10, 0)),
                     text_color='Black')],
            [sg.Text(text="type",
                     font=SliderFont,
                     background_color='#FDF6E3',
                     pad=((0, 0), (10, 0)),
                     text_color='Black'),
             sg.Slider(range=(0, 2), orientation='h', size=(34, 20),
                       default_value=0,
                       background_color='#FDF6E3',
                       text_color='Black',
                       key='-Slider2-',
                       enable_events=True,
                       disable_number_display=True)],
          # pad ((left, right), (top, bottom))
          [sg.Button('Exit', font=AppFont, pad=((540, 0), (0, 0)))]]

_VARS['window'] = sg.Window('Random Samples',
                            layout,
                            finalize=True,
                            resizable=True,
                            location=(100, 100),
                            element_justification="center",
                            background_color='#FDF6E3')

# \\  -------- PYSIMPLEGUI -------- //


# \\  -------- PYPLOT -------- //


fig = plt.figure(figsize = (20,10))



def drawChart():
    ploter(_VARS['i'],_VARS['j'],fig)
    _VARS['pltFig'] = fig
    _VARS['fig_agg'] = draw_figure(
        _VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])


def updateChart():
    _VARS['fig_agg'].get_tk_widget().forget()
    #fig.cla()
    #fig.clf()
    for a in fig.get_axes():
        a.remove()
    ploter(_VARS['i'],_VARS['j'],fig)
    _VARS['pltFig'] = fig
    _VARS['fig_agg'] = draw_figure(
        _VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])


def updateData_i(val):
    _VARS['i'] = val
    updateChart()

def updateData_j(val):
    _VARS['j'] = val
    updateChart()

# \\  -------- PYPLOT -------- //

fig= plt.figure(figsize = (28,13))
drawChart()

# MAIN LOOP
while True:
    event, values = _VARS['window'].read(timeout=200)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    elif event == 'Resample':
        updateChart()
    elif event == '-Slider1-':
        updateData_i(int(values['-Slider1-']))
    elif event == '-Slider2-':
        updateData_j(int(values['-Slider2-']))
        # print(values)
        # print(int(values['-Slider-']))
_VARS['window'].close()
