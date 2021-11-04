# %% markdown
# _italic text_
# **bold text**
# <markdown>
# - Indented
#     - Lists


# **bold text**

# %% md
# **imports**
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


# %%
find_pos = lambda x : np.abs(x) if x>0 else np.nan
find_neg = lambda x : np.abs(x) if x<0 else np.nan
# %%

#source = 'flash'
source = 'stroke'
# %%
dfs = [pd.read_csv(f'data/data_2000-2004_Impact_System/{source}data_200{i}.asc',sep='|',names=['num','date','nano','st_y','st_x','amplitude','icloud','flash'],index_col='date',parse_dates=['date']) for i in range(5)]
df1 = pd.concat(dfs,join='inner',ignore_index=False)
#fig,ax = render_mpl_table(df1, header_columns=0, col_width=2.0)
df1['amp_abs'] = np.abs(df1['amplitude'])
df1['amp_pos'] = df1['amplitude'].apply(find_pos)
df1['amp_neg'] = df1['amplitude'].apply(find_neg)
df1
# %%
dfs = [pd.read_csv(f'data/data_2010-2014_LS7000_System/{source}data_201{i}.asc',sep='|',names=['num','date','nano','st_y','st_x','amplitude','icloud','flash'],index_col='date',parse_dates=['date']) for i in range(5)]
df2 = pd.concat(dfs,join='inner',ignore_index=False)

df2['amp_abs'] = np.abs(df2['amplitude'])
df2['amp_pos'] = df2['amplitude'].apply(find_pos)
df2['amp_neg'] = df2['amplitude'].apply(find_neg)
df2
# %%
dfs = [pd.read_csv(f'data/data_2016-2020_LS7002_System/{source}data_20{i}.asc',sep='|',names=['num','date','nano','st_y','st_x','amplitude','icloud','flash'],index_col='date',parse_dates=['date']) for i in range(16,21)]
df3 = pd.concat(dfs,join='inner',ignore_index=False)

df3['amp_abs'] = np.abs(df3['amplitude'])
df3['amp_pos'] = df3['amplitude'].apply(find_pos)
df3['amp_neg'] = df3['amplitude'].apply(find_neg)
df3

# %%
#general statistics
vals = np.empty((3,2,2))
vals.shape
for i,df in enumerate([df1,df2,df3]):
    l1 = len(df)
    df = df[df['icloud']=='f']
    l2 = len(df)
    lp = len(df[df['amplitude']>0])
    ln = len(df[df['amplitude']<0])
    print(l1,l2,lp,ln)
    vals[i,0,0] = lp
    vals[i,0,1] = ln

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
size = 0.5
#vals = np.array([[60., 32.], [37., 40.], [29., 10.]])
fig,ax = plt.subplots(figsize=(15,15))
cmap = plt.get_cmap("tab20c")
#outer_colors = cmap(np.arange(3)*4)
#inner_colors = cmap([1,2,5,6,9,10])

inner_colors = cmap([0,4,8])
middle_colors = cmap([0,2,4,5,8,9])
outer_colors = cmap([0,1,2,3,4,5,6,7,8,9,10,11])

ax.pie(vals.sum(axis=(1,2)), radius=1-size, colors=inner_colors,wedgeprops=dict(width=1-size, edgecolor='w'),labels=['2000_2004','2010-2014','2016-2020'],labeldistance=.3,rotatelabels=True,explode=None,autopct='%1.1f%%',pctdistance=1)
ax.pie(vals.sum(axis=2).flatten(), radius=1, colors=middle_colors,wedgeprops=dict(width=size, edgecolor='w'),labels=['cloud-to-ground','intracloud']*3,labeldistance=.6,rotatelabels=True,explode=None,autopct='%1.1f%%',pctdistance=1)
ax.pie(vals.flatten(), radius=1+size, colors=outer_colors,wedgeprops=dict(width=size, edgecolor='w'),labels=['positive','negative']*6,labeldistance=.75,rotatelabels=True,explode=None,autopct='%1.1f%%',pctdistance=1)

ax.set(aspect="equal", title='')
plt.show()
fig.savefig('saved_images/statistics_percent.png',dpi=300)
#fig.savefig('saved_images/statistics.png',dpi=10*fig.dpi)



# %%
size = 0.5
#vals = np.array([[60., 32.], [37., 40.], [29., 10.]])
fig,ax = plt.subplots(figsize=(15,15))
cmap = plt.get_cmap("tab20c")
#outer_colors = cmap(np.arange(3)*4)
#inner_colors = cmap([1,2,5,6,9,10])

inner_colors = cmap([0,4,8])
middle_colors = cmap([0,2,4,5,8,9])
outer_colors = cmap([0,1,2,3,4,5,6,7,8,9,10,11])

ax.pie(vals.sum(axis=(1,2)), radius=1-size, colors=inner_colors,wedgeprops=dict(width=1-size, edgecolor='w'),labels=['2000_2004','2010-2014','2016-2020'],labeldistance=.3,rotatelabels=True,explode=None,autopct=lambda x: '{:.0f}'.format(x*vals.sum()/100),pctdistance=1)
ax.pie(vals.sum(axis=2).flatten(), radius=1, colors=middle_colors,wedgeprops=dict(width=size, edgecolor='w'),labels=['cloud-to-ground','intracloud']*3,labeldistance=.6,rotatelabels=True,explode=None,autopct=lambda x: '{:.0f}'.format(x*vals.sum()/100),pctdistance=1)
ax.pie(vals.flatten(), radius=1+size, colors=outer_colors,wedgeprops=dict(width=size, edgecolor='w'),labels=['positive','negative']*6,labeldistance=.75,rotatelabels=True,explode=None,autopct=lambda x: '{:.0f}'.format(x*vals.sum()/100),pctdistance=1)

ax.set(aspect="equal", title='')
plt.show()
fig.savefig('saved_images/statistics_value.png',dpi=300)


# %%
# Effect of lightning type
for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
    #ax = benford(list(df['amp_abs'].groupby([df.index.year,df.index.dayofyear]).count()),f'all lightnings types in day {i}')
    f = benford(list(df['amp_abs'].groupby(df.index.floor('d')).count()),f'number of {source}{"s" if source=="stroke" else "es"} per day in {i}')
    plt.xticks(range(1,10))
    plt.xlabel("first digit")
    plt.ylabel("probability")
    #f.savefig(f'all_{i}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)
# %%
for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
    f = benford(list(df['amp_pos'].groupby(df.index.floor('d')).count()),f'number of positive {source}{"s" if source=="stroke" else "es"} per day in {i}')
    plt.xticks(range(1,10))
    plt.xlabel("first digit")
    plt.ylabel("probability")
    #f.savefig(f'pos_{i}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)
# %%
for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
    f = benford(list(df['amp_neg'].groupby(df.index.floor('d')).count()),f'number of negative {source}{"s" if source=="stroke" else "es"} per day in {i}')
    plt.xticks(range(1,10))
    plt.xlabel("first digit")
    plt.ylabel("probability")
    #f.savefig(f'neg_{i}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)




# %%
D = [[0,0,0],[0,0,0],[0,0,0]]

years = ['','2000-2004','2010-2014','2016-2020','']
polarity = ['all','positive','negative']
for i,df in zip([0,1,2],[df1,df2,df3]):
    for j,col in zip([0,1,2],['amp_abs','amp_pos','amp_neg']):
        jsd, kld, inf, sup, num = distance(list(df[col].groupby(pd.Grouper(freq='1D')).count()))
        D[i][j] = jsd


lns = []
fig, ax = plt.subplots(figsize = (10,5))
for i,c in zip([0,1,2],['r-o','b-o','g-o']):
    ln, = ax.plot([0,1,2],[D[temp][i] for temp in [0,1,2]],c,label=polarity[i]+f' {source}{"s" if source=="stroke" else "es"}')
    lns.append(ln)
    plt.xticks([-1,0,1,2,3],years)
    plt.title(f'{source}{"s" if source=="stroke" else "es"} per day')
ax.legend(handles=lns, loc='best')
ax.set_xlabel("time window")
ax.set_ylabel("JSD")
#fig.savefig(f'compare.png',dpi=10*fig.dpi,bbox_inches='tight',pad_inches=.5)
# %%
D = [[0,0,0],[0,0,0],[0,0,0]]

years = ['','2000-2004','2010-2014','2016-2020','']
polarity = ['all','positive','negative']
for i,df in zip([0,1,2],[df1,df2,df3]):
    df = df[df['icloud']=='f']
    for j,col in zip([0,1,2],['amp_abs','amp_pos','amp_neg']):
        jsd, kld, inf, sup, num = distance(list(df[col].groupby(pd.Grouper(freq='1D')).count()))
        D[i][j] = jsd


lns = []
fig, ax = plt.subplots(figsize = (10,5))
for i,c in zip([0,1,2],['r-o','b-o','g-o']):
    ln, = ax.plot([0,1,2],[D[temp][i] for temp in [0,1,2]],c,label=polarity[i]+f' {source}{"s" if source=="stroke" else "es"}')
    lns.append(ln)
    plt.xticks([-1,0,1,2,3],years)
    plt.title(f'cloud-to-ground {source}{"s" if source=="stroke" else "es"} per day')
ax.legend(handles=lns, loc='best')
ax.set_xlabel("time window")
ax.set_ylabel("JSD")
#fig.savefig(f'compare_ctg.png',dpi=10*fig.dpi,bbox_inches='tight',pad_inches=.5)


# %%
# Effect of counting period
for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
    f = benford(list(df['amp_abs'].groupby(pd.Grouper(freq='5D')).count()),f'number of cloud-to-ground {source}{"s" if source=="stroke" else "es"} per day in {i}')
    plt.xticks(range(1,10))
    plt.xlabel("first digit")
    plt.ylabel("probability")
# %%
for freq in ['1H','6H','12H','24H','48H','96H','7D','1M']:
    df = df1
    f = benford(list(df['amp_abs'].groupby(pd.Grouper(freq=freq)).count()),f'number of {source}{"s" if source=="stroke" else "es"} per {freq}')
    plt.xticks(range(1,10))
    plt.xlabel("first digit")
    plt.ylabel("probability")
    #f.savefig(f'{freq}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)













# %%
x = np.logspace(1,3,100).astype('uint')
y = []
sups = []
infs = []
nums = []
for i in x:
    jsd, kld, inf, sup, num = distance(list(df1['amp_abs'].groupby(pd.Grouper(freq=f'{i}H')).count()))
    y.append(jsd)
    sups.append(sup)
    infs.append(inf)
    nums.append(num)
fig, ax = plt.subplots(figsize = (20,10))
part1 = ax.twinx()
part2 = ax.twinx()
part3 = ax.twinx()


p0, = ax.plot(x,y,label='JSD')
ax.set_xlabel('time slice (hour)')
ax.set_ylabel('jensen shannon distance')

p1, = part1.plot(x,sups,'r',label='maximum')
part1.set_ylabel('maximum')

p2, = part2.plot(x,infs,'g',label='minimum')
part2.set_ylabel('minimum')

p3, = part3.plot(x,nums,'black',label='number of time slices')
part3.set_ylabel('number of time slices')


#plt.xticks(x)
plt.xscale('log')
lns = [p0,p1,p2,p3]
ax.legend(handles=lns, loc='upper center')


# right, left, top, bottom
part2.spines['right'].set_position(('outward', 90))
part3.spines['left'].set_position(('outward',90))
part3.spines['left'].set_visible(True)
part3.yaxis.set_label_position('left')
part3.yaxis.set_ticks_position('left')

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
part3.yaxis.label.set_color(p3.get_color())

#fig.tight_layout()
#plt.title(f"simulation with {distribution} interarrival time")
#fig.savefig(f'out.png',dpi=10*fig.dpi,bbox_inches='tight',pad_inches=1)
#dict[f'{distribution}'] = [x,y,sups,infs]










# %%
#x = np.logspace(1,3,500).astype('uint')
x = np.logspace(0,3,500)
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
        y_t[i].append([])
        sups_t[i].append([])
        infs_t[i].append([])
        nums_t[i].append([])
        for freq in x:
            freq = freq.astype('uint')
            jsd, kld, inf, sup, num = distance(list(df[col_name].groupby(pd.Grouper(freq=f'{freq}H')).count()))
            y_t[i][j].append(jsd)
            sups_t[i][j].append(sup)
            infs_t[i][j].append(inf)
            nums_t[i][j].append(num)


# # %%
# fig,ax = plt.subplots(figsize = (20,14))
# for i in range(3):
#     for j in range(3):
#         ax.plot(x,y_t[i][j])
#
# #plt.axvline(x=24, ymin=0.05, ymax=0.2, color='b', label='axvline - % of full height')
# #plt.axvline(x=120, ymin=0.05, ymax=0.2, color='b', label='axvline - % of full height')
# plt.axvline(x=72, ymin=0.045, ymax=0.32, color='g', label='axvline - % of full height')
# #plt.axhline(y=0.05, xmin=0.072, xmax=.120, color='b', label='axvline - % of full height')
# #plt.axhline(y=0.08, xmin=0.072, xmax=.120,  color='b', label='axvline - % of full height')
#
# plt.gca().add_patch(mpatches.Rectangle(xy=[24, 0.05], width=96, height=0.04,facecolor='None',edgecolor='red',linewidth=3,alpha=0.9))
# plt.xscale('log')







# %%
year = ['2000-2004','2010-2014','2016-2020']
polarity = ['all','positive','negative']
i = 0
j = 0
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
ax.set_xlabel('time slice (hours)')

p1, = part1.plot(x,y,'C0',label='JSD',alpha=0.5)

part1.set_ylabel('Jensen Shannon Distance')

p2, = part2.plot(x,sups,'r',label='maximum',alpha=1)
part2.set_ylabel('maximum')

#p3, = part3.plot(x,infs,'g',label='minimum',alpha=1)
#part3.set_ylabel('minimum')

#plt.xticks(x)
plt.xscale('log')

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



plt.xscale('log')

#fig.savefig(f'out_{i}{j}.png',dpi=10*fig.dpi,bbox_inches='tight',pad_inches=1)


# 300 represents number of points to make between T.min and T.max
#xnew = np.logspace(np.log10(x.min()), np.log10(x.max()), 1000)

#spl = make_interp_spline(np.log10(x), y, k=5)  # type: BSpline
#y_smooth = spl(np.log10(xnew))



kernel_size = 30
kernel = np.ones(kernel_size) / kernel_size
y_smooth = np.convolve(y, kernel, mode='same')
y_smooth = np.convolve(np.pad(y,(kernel_size//2,kernel_size-kernel_size//2-1),mode='edge'), kernel, mode='valid')

p3, = part1.plot(x,y_smooth,'C0',label='JSD-smooth',alpha=1)

p4 = part1.add_patch(mpatches.Rectangle(xy=[24, 0.05], width=96, height=0.04,facecolor='None',edgecolor='green',linewidth=3,alpha=0.9))
#fig.savefig(f'saved_images/out_{i}{j}.png',dpi=10*fig.dpi,bbox_inches='tight',pad_inches=1)
p1.set_alpha(0.5)
p2.set_alpha(1)
p3.set_alpha(1)
p4.set_alpha(1)
lns = [p0,p1,p3,p2]
ax.legend(handles=lns, loc='upper center')
fig.savefig(f'saved_images/step_5.png',dpi=300,bbox_inches='tight')


# %%
year = ['2000-2004','2010-2014','2016-2020']
polarity = ['all','positive','negative']
#fig,a = plt.subplots(3,3,figsize = (20,10))

f,a = plt.subplots(1,1,figsize=(25,14))
#plt.title(f'perfect window')
a.plot([0,1],[0,1],'r-*',alpha=0)
plt.xticks([0.13,0.50,0.88],[f'all {source}{"s" if source=="stroke" else "es"}',f'positive {source}{"s" if source=="stroke" else "es"}',f'negative {source}{"s" if source=="stroke" else "es"}'])
plt.yticks([0.13,0.50,0.87],['2016-2020','2010-2014','2000_2004'])
plt.ylabel('time window')
plt.xlabel('lightning type')


for i in range(3):
    for j in range(3):

        y = y_t[i][j]
        sups = sups_t[i][j]
        infs = infs_t[i][j]
        nums = nums_t[i][j]

        left = 0.14 + j * 0.255
        bottom = 0.64 - i * 0.25
        width = 0.24
        height = 0.22
        right = left + width
        top = bottom + height
        center_lr = (left + right)/2
        center_tb = (top+bottom)/2
        #a.plot([left,left,right,right],[top,bottom,top,bottom],'ro')
        #a.plot([center_lr],[center_tb],"b*",transform=a.transAxes)
        ax = plt.axes([left,bottom,width,height])
        #ax = a[i][j]


        part1 = ax.twinx()
        part2 = ax.twinx()
        #part3 = ax.twinx()

        p0, = ax.plot(x,nums,'black',label='number of time slices')
        ax.set_ylabel('number of time slices')
        ax.set_xlabel('time slice (hour)')

        p1, = part1.plot(x,y,'C0',label='JSD',alpha=0.5)

        part1.set_ylabel('Jensen Shannon Distance')

        p2, = part2.plot(x,sups,'r',label='maximum',alpha=1)
        part2.set_ylabel('maximum')

        #p3, = part3.plot(x,infs,'g',label='minimum',alpha=1)
        #part3.set_ylabel('minimum')

        #plt.xticks(x)
        plt.xscale('log')

        #lns = [p0,p1,p2,p3]



        # right, left, top, bottom
        #ax.spines['right'].set_visible(False)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        part1.spines['right'].set_visible(False)
        part1.yaxis.set_visible(False)
        part2.spines['right'].set_position(('outward', 90))
        part2.spines['right'].set_visible(False)
        part2.yaxis.set_visible(False)
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



        plt.xscale('log')

        #fig.savefig(f'out_{i}{j}.png',dpi=10*fig.dpi,bbox_inches='tight',pad_inches=1)


        # 300 represents number of points to make between T.min and T.max
        #xnew = np.logspace(np.log10(x.min()), np.log10(x.max()), 1000)

        #spl = make_interp_spline(np.log10(x), y, k=5)  # type: BSpline
        #y_smooth = spl(np.log10(xnew))



        kernel_size = 30
        kernel = np.ones(kernel_size) / kernel_size
        y_smooth = np.convolve(y, kernel, mode='same')
        y_smooth = np.convolve(np.pad(y,(kernel_size//2,kernel_size-kernel_size//2-1),mode='edge'), kernel, mode='valid')

        p3, = part1.plot(x,y_smooth,'C0',label='JSD-smooth',alpha=1)

        p4 = part1.add_patch(mpatches.Rectangle(xy=[24, 0.05], width=96, height=0.04,facecolor='None',edgecolor='green',linewidth=3,alpha=0.9))
        #fig.savefig(f'saved_images/out_{i}{j}.png',dpi=10*fig.dpi,bbox_inches='tight',pad_inches=1)
        p1.set_alpha(0.5)
        p2.set_alpha(1)
        p3.set_alpha(1)
        p4.set_alpha(1)
        lns = [p0,p1,p3,p2]
        #ax.legend(handles=lns, loc='upper center')
f.savefig(f'saved_images/grid_out_window.png',dpi=300,bbox_inches='tight')







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
x = np.logspace(1,3,100).astype('uint')
y = []
sups = []
infs = []
nums = []
for i in x:
    jsd, kld, inf, sup, num = distance(list(df1['amp_abs'].groupby(pd.Grouper(freq=f'{i}H')).count()))
    y.append(jsd)
    sups.append(sup)
    infs.append(inf)
    nums.append(num)
# %%
fig, ax = plt.subplots(figsize = (20,10))
part1 = ax.twinx()
part2 = ax.twinx()
#part3 = ax.twinx()

p0, = ax.plot(x,nums,'black',label='number of time slices')
ax.set_ylabel('number of time slices')
ax.set_xlabel('time slice (hour)')

p1, = part1.plot(x,y,'C0',label='JSD',alpha=1)
part1.set_ylabel('Jensen Shannon Distance')

p2, = part2.plot(x,sups,'r',label='maximum',alpha=1)
part2.set_ylabel('maximum')

#p3, = part3.plot(x,infs,'g',label='minimum',alpha=1)
#part3.set_ylabel('minimum')

#plt.xticks(x)
plt.xscale('log')

#lns = [p0,p1,p2,p3]
lns = [p0,p1,p2]
ax.legend(handles=lns, loc='upper right')


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
#plt.title(f"simulation with {distribution} interarrival time")
#fig.savefig(f'out_3.png',dpi=10*fig.dpi,bbox_inches='tight',pad_inches=1)

















# # %%
# # Effect of lightning destination
# for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
#     df = df[df['icloud']=='f']
#     f = benford(list(df['amp_abs'].groupby(df.index.floor('d')).count()),f'number of cloud-to-ground {source}{"s" if source=="stroke" else "es"} per day in {i}')
#     #f.savefig(f'all_{i}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)
# # %%
# for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
#     df = df[df['icloud']=='f']
#     f = benford(list(df['amp_pos'].groupby(df.index.floor('d')).count()),f'number of positive cloud-to-ground {source}{"s" if source=="stroke" else "es"} per day in {i}')
#     #f.savefig(f'pos_{i}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)
# # %%
# for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
#     df = df[df['icloud']=='f']
#     f = benford(list(df['amp_neg'].groupby(df.index.floor('d')).count()),f'number of negative cloud-to-ground {source}{"s" if source=="stroke" else "es"} per day in {i}')
#     #f.savefig(f'neg_{i}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)
# # %%
# for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
#     df = df[df['icloud']=='t']
#     f = benford(list(df['amp_abs'].groupby(df.index.floor('d')).count()),f'number of cloud-to-ground {source}{"s" if source=="stroke" else "es"} per day in {i}')
#     #f.savefig(f'all_{i}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)
# # %%
# for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
#     df = df[df['icloud']=='t']
#     f = benford(list(df['amp_pos'].groupby(df.index.floor('d')).count()),f'number of positive cloud-to-ground {source}{"s" if source=="stroke" else "es"} per day in {i}')
#     #f.savefig(f'pos_{i}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)
# # %%
# for i,df in zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3]):
#     df = df[df['icloud']=='t']
#     f = benford(list(df['amp_neg'].groupby(df.index.floor('d')).count()),f'number of negative cloud-to-ground {source}{"s" if source=="stroke" else "es"} per day in {i}')
#     #f.savefig(f'neg_{i}.png',dpi=f.dpi,bbox_inches='tight',pad_inches=.5)





# %%
f,a = plt.subplots(1,1,figsize=(25,14))
plt.title(f'{source}{"s" if source=="stroke" else "es"} per day')
a.plot([0,1],[0,1],'r-*',alpha=0)
plt.xticks([0.15,0.51,0.875],[f'all {source}{"s" if source=="stroke" else "es"}',f'positive {source}{"s" if source=="stroke" else "es"}',f'negative {source}{"s" if source=="stroke" else "es"}'])
plt.yticks([0.135,0.505,0.875],['2016-2020','2010-2014','2000_2004'])
plt.ylabel('time window')
plt.xlabel('lightning type')
for j,(polarity,p_text,p_save,sweep) in enumerate(zip(['abs','pos','neg'],['','positive ','negative '],['all','pos','neg'],[(300,30000),(100,10000),(300,30000)])):
    for i,(name,df) in enumerate(zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3])):
        df = df[df['icloud']=='f']
        left = 0.155 + j * 0.255
        bottom = 0.65 - i * 0.25
        width = 0.22
        height = 0.18
        right = left + width
        top = bottom + height
        center_lr = (left + right)/2
        center_tb = (top+bottom)/2
        #a.plot([left,left,right,right],[top,bottom,top,bottom],'ro')
        #a.plot([center_lr],[center_tb],"b*",transform=a.transAxes)
        ax = plt.axes([left,bottom,width,height])
        benford(list(df[f'amp_{polarity}'].groupby(pd.Grouper(freq=f'1D')).count()),f'',ax=ax)
        ax.axes.set_xticks(range(1,10))
        #benford(list(df[f'amp_{polarity}'].groupby(pd.Grouper(freq=f'{freq}D')).count()),f'',limit=limit,ax=ax)
        #numbers = list(df[f'amp_{polarity}'].groupby(pd.Grouper(freq=f'1D')).count())


    #a.legend(handles=[p0,p1,p2],loc = 'upper left')
    f.savefig(f'saved_images/grid_results.png',dpi=300,bbox_inches='tight')








# %%
for polarity,p_text,p_save,limit in zip(['abs','pos','neg'],['','positive ','negative '],['all','pos','neg'],[10000,1000,10000]):
    f,a = plt.subplots(1,1,figsize=(25,14))
    plt.title(f'number of {p_text}cloud-to-ground {source}{"s" if source=="stroke" else "es"} less than {limit}')
    a.plot([0,1],[0,1],'r-*',alpha=0)
    plt.xticks([0.14,0.51,0.88],[1,2,3])
    plt.yticks([0.11,0.48,0.85],['2016-2020','2010-2014','2000_2004'])
    plt.ylabel('time window')
    plt.xlabel('time slice size (day)')
    for i,(name,df) in enumerate(zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3])):
        df = df[df['icloud']=='f']
        for j,freq in enumerate(range(1,4)):
            left = 0.15 + j * 0.26
            bottom = 0.65 - i * 0.25
            width = 0.22
            height = 0.18
            right = left + width
            top = bottom + height
            center_lr = (left + right)/2
            center_tb = (top+bottom)/2
            #a.plot([left,left,right,right],[top,bottom,top,bottom],'ro')
            #a.plot([center_lr],[center_tb],"b*",transform=a.transAxes)
            ax = plt.axes([left,bottom,width,height])
            benford(list(df[f'amp_{polarity}'].groupby(pd.Grouper(freq=f'{freq}D')).count()),f'',limit=limit,ax=ax)
            ax.axes.set_xticks(range(1,10))


    #f.savefig(f'grid_benford_limit_{p_save}.png',dpi=10*f.dpi,bbox_inches='tight',pad_inches=.5)



# %%
for polarity,p_text,p_save,sweep in zip(['abs','pos','neg'],['','positive ','negative '],['all','pos','neg'],[(300,30000),(100,10000),(300,30000)]):
    f,a = plt.subplots(1,1,figsize=(25,14))
    plt.title(f'Thresholding effect, {p_text}cloud-to-ground {source}{"s" if source=="stroke" else "es"}')
    a.plot([0,1],[0,1],'r-*',alpha=0)
    plt.xticks([0.12,0.48,0.85],[1,2,3])
    plt.yticks([0.135,0.505,0.875],['2016-2020','2010-2014','2000_2004'])
    plt.ylabel('time window')
    plt.xlabel('time slice size (day)')
    for i,(name,df) in enumerate(zip(['2000-2004','2010-2014','2016-2020'],[df1,df2,df3])):
        df = df[df['icloud']=='f']
        for j,freq in enumerate(range(1,4)):
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
            numbers = list(df[f'amp_{polarity}'].groupby(pd.Grouper(freq=f'{freq}D')).count())
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
    #f.savefig(f'saved_images/grid_limit_{p_save}.png',dpi=300)









# %%
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

    f.savefig(f'saved_images/grid_benford_limit_{p_save}.png',dpi=300,bbox_inches='tight')


# %%
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

    f.savefig(f'saved_images/grid_benford_limit_multiplier_{p_save}.png',dpi=300,bbox_inches='tight')
# %%
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

    f.savefig(f'saved_images/grid_benford_limit_multiplier_2_times_{p_save}.png',dpi=300,bbox_inches='tight')

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
    #f.savefig(f'saved_images/grid_limit.png',dpi=300)








# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%
x = np.linspace(1,10,10).astype('uint')
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
        y_t[i].append([])
        sups_t[i].append([])
        infs_t[i].append([])
        nums_t[i].append([])
        for freq in x:
            jsd, kld, inf, sup, num = distance(list(df[col_name].groupby(pd.Grouper(freq=f'{freq}D')).count()))
            y_t[i][j].append(jsd)
            sups_t[i][j].append(sup)
            infs_t[i][j].append(inf)
            nums_t[i][j].append(num)




# %%
for i in range(3):
    for j in range(3):
        year = ['2000-2004','2010-2014','2016-2020']
        polarity = ['all','positive','negative']
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
        ax.set_xlabel('time slice (day)')

        p1, = part1.plot(x,y,'C0',label='JSD',alpha=1)
        part1.set_ylabel('Jensen Shannon Distance')

        p2, = part2.plot(x,sups,'r',label='maximum',alpha=1)
        part2.set_ylabel('maximum')

        #p3, = part3.plot(x,infs,'g',label='minimum',alpha=1)
        #part3.set_ylabel('minimum')

        #plt.xticks(x)


        #lns = [p0,p1,p2,p3]
        lns = [p0,p1,p2]
        ax.legend(handles=lns, loc='upper center')


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

        plt.title(f"{polarity[j]} cloud to ground strokes in {year[i]}")


        #part1.axvline(x=72, ymin=0.045, ymax=0.32, color='g', label='axvline - % of full height')


        #part1.add_patch(mpatches.Rectangle(xy=[24, 0.05], width=96, height=0.04,facecolor='None',edgecolor='red',linewidth=3,alpha=0.9))


        #fig.savefig(f'out_D_{i}{j}.png',dpi=10*fig.dpi,bbox_inches='tight',pad_inches=1)
