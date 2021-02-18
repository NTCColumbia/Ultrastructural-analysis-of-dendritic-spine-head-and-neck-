import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os.path
from fnc import *
from scipy.stats import spearmanr
from scipy import stats
from scipy.stats import linregress
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from scipy.stats import ks_2samp


#%% Data Kasthuri
df_K = pd.read_csv('C:/Data_Spines/Kasthuri_spines/Kasthuri_data_new9.txt', sep=" ", header=None)
ind = []
for i in range(len(df_K)):
    ind.append(df_K[0][i])

df_K.index = ind
df_K = df_K.drop([0], axis=1)
df_K.columns = ["Volume", "Area", "Head-Volume", "Head-Area" , "Length" , "Neck-Length", "Neck-Radius" ]

df_K = df_K[["Volume", "Area", "Length" , "Head-Volume", "Head-Area" , "Neck-Length", "Neck-Radius"]]
#import collections ## checking for doubles:
#print([item for item, count in collections.Counter(df_K.index).items() if count > 1])


df_K = df_K[df_K['Length'] < 10]  # removing one case of length=70  --> 3137
df_K = df_K[df_K['Neck-Length'] != df_K['Length']] #                --> 3033
df_K = df_K[df_K['Neck-Radius'] != 0] #                             --> 2998

# removing outliers
z = np.abs(stats.zscore(np.log10(df_K)))
df_K2 = df_K[(z < 3).all(axis=1)] #                                  --> (2904 for z=3)


## log + Z-Score:
df_z = np.log10(df_K2)
for k in list(df_z):
  df_z[k] = ( df_z[k].tolist() - np.mean(df_z[k].tolist()) ) / np.std(df_z[k].tolist())



## correlation
#pearsoncorr = df_z.corr(method='pearson')
#spearmancorr = df_z.corr(method='spearman')
#fig = plt.figure()
#sns.heatmap(pearsoncorr, xticklabels=pearsoncorr.columns, yticklabels=pearsoncorr.columns, cmap='RdBu_r', annot=True, linewidth=0.5)
#plt.tight_layout()
#fig = plt.figure()
#sns.heatmap(spearmancorr, xticklabels=pearsoncorr.columns, yticklabels=pearsoncorr.columns, cmap='RdBu_r', annot=True, linewidth=0.5)
#plt.tight_layout()
#plt.show()


df = df_z.drop(['Neck-Length', 'Head-Volume','Head-Area','Neck-Radius'], axis=1) #'Area', 

### Grid 1
mpl.rcParams["axes.labelsize"] = 16
mpl.rcParams['figure.figsize']=(11,10)
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
grid = sns.PairGrid(data=df)
grid = grid.map_upper(sns.scatterplot, color = 'darkred', alpha=0.1,edgecolor = None, s=2) 
grid = grid.map_diag(plt.hist, bins = 50, color = 'darkred', edgecolor = 'k') # 100
grid = grid.map_lower(sns.kdeplot, cmap = 'Reds', n_levels=10) # 20
plt.annotate('A', xy=(5, 520), xycoords='figure points', fontsize=16)
plt.tight_layout()
fileName3 = ("C:/Data_Spines/Kasthuri_spines/figures_paper2/Grid1.pdf" )
#grid.savefig(fileName3)
plt.show()


####### Head and Neck analysis
df = df_z.drop(['Area', 'Volume', 'Head-Area', 'Length'], axis=1) # 

### Grid 2
mpl.rcParams["axes.labelsize"] = 16
mpl.rcParams['figure.figsize']=(11,10)
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
grid2 = sns.PairGrid(data=df)
grid2 = grid2.map_upper(sns.scatterplot, color = 'darkred', alpha=0.1,edgecolor = None, s=2) 
grid2 = grid2.map_diag(plt.hist, bins = 50, color = 'darkred', edgecolor = 'k')
grid2 = grid2.map_lower(sns.kdeplot, cmap = 'Reds', n_levels=10)
plt.annotate('B', xy=(5, 520), xycoords='figure points', fontsize=16)
plt.tight_layout()
fileName3 = ("C:/Data_Spines/Kasthuri_spines/figures_paper2/Grid2.pdf" )
#grid.savefig(fileName3)
plt.show()



### Sphericity
sph = (np.pi**(1/3.0))*(6*df_K['Head-Volume'])**(2/3) / df_K['Head-Area']
sph = sph[sph<1]
df_K['sphericity'] = sph
df_KK = df_K



##############

Kasthuri2015_mmc6 = pd.read_excel (r'C:\Data_Spines\kasthuri2015_mmc6.xls')

### spines ID that included in the table: (#1198)
spine = []
PSD = []
Apparatus = []
for k in range(1,len(Kasthuri2015_mmc6['Spine ID No.'])):
    if (Kasthuri2015_mmc6['Spine ID No.'][k] > 0):
        spine.append(int(Kasthuri2015_mmc6['Spine ID No.'][k]))
        PSD.append(int(Kasthuri2015_mmc6['PSD size'][k]))
        Apparatus.append(int(Kasthuri2015_mmc6['Spine Apparatus'][k])) # NEW Spine Apparatus , No = 0; Yes = 1 , N/A = -1 , Uncertain = -2

ss = {'PSD':PSD , 'Apparatus':Apparatus}
Spines = pd.DataFrame(ss)
Spines.index = spine
# str(int(spine[0])).zfill(4) # there are two 5140!!
Spines = Spines.drop(5140, axis=0) # throwing the doubles.


ind = []
for i in range(len(df_K)):
    ind.append(int(df_K.index[i].split("_")[2]))

df_K.index = ind

### Apparatus Vs. type:
df2 = pd.merge(Spines, df_K, left_index=True, right_index=True)



##########################################


### plot all histograms - step!
Let = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
xlbl = [r'Spine volume ($\mu m^{3}$)', r'Spine surface area ($\mu m^2$)', r'Spine length ($\mu m$)', r'Head volume ($\mu m^{3}$)', 
        r'Neck length ($\mu m$)', r'Neck radius ($\mu m$)', 'Sphericity']
ll = ['Volume', 'Area', 'Length', 'Head-Volume', 'Neck-Length', 'Neck-Radius', 'sphericity'] # 
fig = plt.figure(figsize=(15,7.5))
for ch in range(len(list(df_KK))-1):
    rng = [np.min(df_KK[list(df_KK)[ch]]) , np.max(df_KK[list(df_KK)[ch]])]
    ax = fig.add_subplot(2, len(list(df_KK))/2, ch+1 )
    ar = np.array(df_KK[ll[ch]] ) #
    ax.hist(ar, bins = 35, histtype='step', color='b', density=False )
    ax.set_xlabel(xlbl[ch], fontsize=16) # 
    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.annotate(Let[ch], xy=(-0.3, 1), xycoords='axes fraction', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
ax.set_xlim([0.45,1])
ax4 = fig.add_subplot(2, len(list(df_KK))/2, 8)
ar = np.array(df2['PSD'])
ax4.hist(ar, bins = 50, histtype='step', color='b' )
ax4.set_xlabel('PSD size (pixels)', fontsize=16)
ax4.tick_params(axis='x', which='major', labelsize=16)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.annotate(Let[7], xy=(-0.3, 1), xycoords='axes fraction', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
fileName3 = ("C:/Data_Spines/Kasthuri_spines/figures_paper2/Histograms5.pdf" )
#fig.savefig(fileName3)
plt.show()







## Figure Correlation:
cl_corr = ['b', 'y', 'r']
fig = plt.figure(figsize=(16,5))
for pp in range(4):
    ax = fig.add_subplot(1, 4, pp+1)
    if (pp==0):
        x = df_KK['Neck-Radius']*2; y = df_KK['Neck-Length']
        #x = np.log10(x); y = np.log10(y)
        ax.set_xlabel(r'Neck diameter ($\mu m$)', fontsize=16)
        ax.set_ylabel(r'Neck length ($\mu m$)', fontsize=16)
        ax.set_ylim([np.min(df_K['Neck-Length'])*0.9,np.max(df_K['Neck-Length'])*1.1])
        ax.set_xlim([np.min(df_K['Neck-Radius'])*1.8,np.max(df_K['Neck-Radius'])*2.2])
    if (pp==1):
        x = df_KK['Neck-Radius']*2; y = df_KK['Head-Volume']
        #x = np.log10(x); y = np.log10(y)
        ax.set_xlabel(r'Neck diameter ($\mu m$)', fontsize=16)
        ax.set_ylabel(r'Head volume ($\mu m^{3}$)', fontsize=16)
        ax.set_ylim([np.min(df_K['Head-Volume'])*0.9,np.max(df_K['Head-Volume'])*1.1])
        ax.set_xlim([np.min(df_K['Neck-Radius'])*1.8,np.max(df_K['Neck-Radius'])*2.2])
    if (pp==2):
        x = df_KK['Neck-Length']; y = df_KK['Head-Volume']
        #x = np.log10(x); y = np.log10(y)
        ax.set_xlabel(r'Neck length ($\mu m$)', fontsize=16)
        ax.set_ylabel(r'Head volume ($\mu m^{3}$)', fontsize=16)
        ax.set_ylim([np.min(df_K['Head-Volume'])*0.9,np.max(df_K['Head-Volume'])*1.1])
        ax.set_xlim([np.min(df_K['Neck-Length'])*0.9,np.max(df_K['Neck-Length'])*1.1])
    if (pp==3):
        x = df2['Head-Volume']; y = df2['PSD']
        #x = np.log10(x); y = np.log10(y)
        ax.set_xlabel(r'Head volume ($\mu m^{3}$)', fontsize=16)
        ax.set_ylabel('PSD size (pixels)', fontsize=16)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    ax.scatter(x, y, s=3, color='b', alpha=0.2) # 'b'
    ax.plot(x, intercept + slope*x, 'gray', label='fitted line')
    ax.annotate("r=%.3g" % r_value, xy=(0.04, 0.94), xycoords='axes fraction', fontsize=16) #, ha='center'
    if (p_value<0.001):
        ast = '***'
    elif (p_value<0.01):
        ast = '**'
    elif (p_value<0.05):
        ast = '*'
    else:
        ast = ''
    #
    ax.annotate(ast, xy=(0.5, 0.94), xycoords='axes fraction', fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.94, hspace=0.4)
plt.annotate('A', xy=(0.01, 0.95), xycoords='figure fraction', fontsize=18)
plt.annotate('B', xy=(0.25, 0.95), xycoords='figure fraction', fontsize=18)
plt.annotate('C', xy=(0.5, 0.95), xycoords='figure fraction', fontsize=18) 
plt.annotate('D', xy=(0.74, 0.95), xycoords='figure fraction', fontsize=18)
fileName3 = ("C:/Data_Spines/Kasthuri_spines/figures_paper2/Correlations.pdf" )
#fig.savefig(fileName3)
plt.show()








df3 = df2.copy()

mrs = ['none' , 'r' ,'none','none']
mrc = np.array(mrs)[df3['Apparatus']].tolist()
ers = ['b' , 'none' ,'none','none']
erc = np.array(ers)[df3['Apparatus']].tolist()
kk = [None]*len(df2)
kk[0] = 'With SA'; kk[1] = 'Without SA'

######################## Figure for the paper:
fig = plt.figure(figsize=(12,9))
#ax5 = plt.subplot2grid((2,3), (0,0), colspan=2, rowspan=2)
ax5 = plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
for xp, yp, m, l, kkk in zip(df3['Head-Volume'], df3['Neck-Radius']*2, mrc, erc, kk): # PSD , Neck-Radius
    ax5.scatter([xp],[yp], s=60, facecolors=m , edgecolors=l, alpha=0.8, label=kkk) # , marker='o'c='r', label=clo
ax5.set_xlabel(r'Head volume ($\mu m^{3}$)', fontsize=16)
ax5.set_ylabel(r'Neck diameter ($\mu m$)', fontsize=16)
ax5.tick_params(axis="x", labelsize=16)
ax5.tick_params(axis="y", labelsize=16)
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.spines['right'].set_visible(False)
ax5.spines['top'].set_visible(False)
ax5.legend(loc='upper left', fontsize=16) # 'best'
ax5.annotate('A', xy=(-0.142, 1), xycoords='axes fraction', fontsize=16)

ax4 = fig.add_subplot(3, 3, 3)
sp = np.min(df3['Head-Volume']); ep = np.max(df3['Head-Volume']) # start/end point
ml = [i/len(df3['Head-Volume'][df3['Apparatus']==0]) for i in range(len(df3['Head-Volume'][df3['Apparatus']==0]))]
ml.append(1)
ax4.plot(np.append(np.sort(df3['Head-Volume'][df3['Apparatus']==0]), ep) , ml, 'b')
nl = [i/len(df3['Head-Volume'][df3['Apparatus']==1]) for i in range(len(df3['Head-Volume'][df3['Apparatus']==1]))]
nl.insert(0,0)
ax4.plot(np.insert(np.sort(df3['Head-Volume'][df3['Apparatus']==1]), 0, sp) , nl, 'r')
#ax4.annotate('***', xy=(0.5, 0.90), xycoords='axes fraction', fontsize=16, ha='center') #
ax4.set_xlabel(r'Head volume ($\mu m^{3}$)', fontsize=16)
ax4.set_ylabel('Probability', fontsize=16)
ax4.tick_params(axis="x", labelsize=16)
ax4.tick_params(axis="y", labelsize=16)
ax4.set_xscale('log')
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.annotate('B', xy=(-0.3, 1), xycoords='axes fraction', fontsize=16)

sp = np.min(df3['Neck-Length']) ; ep = np.max(df3['Neck-Length']) # start/end point
ax2 = fig.add_subplot(3, 3, 6)
ml = [i/len(df3['Neck-Length'][df3['Apparatus']==0]) for i in range(len(df3['Neck-Length'][df3['Apparatus']==0]))]
ml.append(1)
ax2.plot(np.append(np.sort(df3['Neck-Length'][df3['Apparatus']==0]), ep) , ml, 'b')
nl = [i/len(df3['Neck-Length'][df3['Apparatus']==1]) for i in range(len(df3['Neck-Length'][df3['Apparatus']==1]))]
nl.insert(0,0)
#ax2.plot(np.sort(df3['Neck-Length'][df3['Apparatus']==1]) , [i/len(df3['Neck-Length'][df3['Apparatus']==1]) for i in range(len(df3['Neck-Length'][df3['Apparatus']==1]))], 'r')
ax2.plot(np.insert(np.sort(df3['Neck-Length'][df3['Apparatus']==1]), 0, sp) , nl, 'r')
ax2.set_xlabel(r'Neck length ($\mu m$)', fontsize=16)
ax2.set_ylabel('Probability', fontsize=16)
ax2.tick_params(axis="x", labelsize=16)
ax2.tick_params(axis="y", labelsize=16)
ax2.set_xscale('log')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.annotate('C', xy=(-0.3, 1), xycoords='axes fraction', fontsize=16)

sp = np.min(df3['Neck-Radius']); ep = np.max(df3['Neck-Radius']) # start/end point
ax3 = fig.add_subplot(3, 3, 9)
ml = [i/len(df3['Neck-Radius'][df3['Apparatus']==0]) for i in range(len(df3['Neck-Radius'][df3['Apparatus']==0]))]
ml.append(1)
ax3.plot(np.append(np.sort(df3['Neck-Radius'][df3['Apparatus']==0])*2, ep*2) , ml, 'b')
nl = [i/len(df3['Neck-Radius'][df3['Apparatus']==1]) for i in range(len(df3['Neck-Radius'][df3['Apparatus']==1]))]
nl.insert(0,0)
ax3.plot(np.insert(np.sort(df3['Neck-Radius'][df3['Apparatus']==1])*2, 0, sp*2) , nl, 'r')
#ax3.annotate('***', xy=(0.5, 0.90), xycoords='axes fraction', fontsize=16, ha='center') #
ax3.set_xlabel(r'Neck diameter ($\mu m$)', fontsize=16)
ax3.set_ylabel('Probability', fontsize=16)
ax3.tick_params(axis="x", labelsize=16)
ax3.tick_params(axis="y", labelsize=16)
ax3.set_xscale('log')
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.annotate('D', xy=(-0.3, 1), xycoords='axes fraction', fontsize=16)

plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
##fileName2 = ("C:/Data_Spines/Kasthuri_spines/figures_paper2/Apparatus.png" )
fileName3 = ("C:/Data_Spines/Kasthuri_spines/figures_paper2/Apparatus.pdf" ) 
##fig.savefig(fileName2, dpi=1500)
#fig.savefig(fileName3)
plt.show()



ks_2samp(df3['Head-Volume'][df3['Apparatus']==0] , df3['Head-Volume'][df3['Apparatus']==1]) # pvalue=1.9984014443252818e-15
ks_2samp(df3['Neck-Length'][df3['Apparatus']==0] , df3['Neck-Length'][df3['Apparatus']==1]) # pvalue=0.3677176448815974
ks_2samp(df3['Neck-Radius'][df3['Apparatus']==0] , df3['Neck-Radius'][df3['Apparatus']==1]) # pvalue=1.9984014443252818e-15




##### Dual connection:

Kasthuri_Connectome = np.zeros((6739,6739))
for i in range(1,len(Kasthuri2015_mmc6['Axon No'])):
    Kasthuri_Connectome[int(Kasthuri2015_mmc6['Axon No'][i]) , int(Kasthuri2015_mmc6['Dendrite No'][i])] +=1

#%% Finding the structures of spines from dual-connections:
duals = list(zip(*np.where(Kasthuri_Connectome==2))) # 107

spine_num = []
spine_num2 = []
for i,j in duals:
    for k in range(1,len(Kasthuri2015_mmc6['Axon No'])):
        if ((Kasthuri2015_mmc6['Axon No'][k]==i) & (Kasthuri2015_mmc6['Dendrite No'][k]==j)):
            spine_num.append(Kasthuri2015_mmc6['Spine ID No.'][k])
    spine_num2.append(spine_num)
    spine_num = []

spine_num3 = []
for s in spine_num2:
    if ((s[0]!=0) & (s[0]!=-1) & (s[1]!=0) & (s[1]!=-1)):
        spine_num3.append(s)

len(spine_num3) # 71





#################### Cumulative spines in duals Vs. all spines

df3 = df_K.copy()

HV = []; NL = []; ND = []
HV_dif = []; NL_dif = []; ND_dif = []
HV_rat = []; NL_rat = []; ND_rat = []
for s in spine_num3:
    if ((s[0] in df2.index) & (s[1] in df2.index)):
        hv1 = df2.loc[int(s[0])]['Head-Volume']
        hv2 = df2.loc[int(s[1])]['Head-Volume']
        nl1 = df2.loc[int(s[0])]['Neck-Length']
        nl2 = df2.loc[int(s[1])]['Neck-Length']
        nd1 = df2.loc[int(s[0])]['Neck-Radius']*2
        nd2 = df2.loc[int(s[1])]['Neck-Radius']*2
        HV.append(hv1); HV.append(hv2)
        NL.append(nl1); NL.append(nl2)
        ND.append(nd1); ND.append(nd2)
        HV_dif.append(np.abs(hv1-hv2))
        NL_dif.append(np.abs(nl1-nl2))
        ND_dif.append(np.abs(nd1-nd2))
        # ratio
        HV_rat.append(np.max([hv1,hv2])/np.min([hv1,hv2]))
        NL_rat.append(np.max([nl1,nl2])/np.min([nl1,nl2]))
        ND_rat.append(np.max([nd1,nd2])/np.min([nd1,nd2]))


hvd = []; nld = []; ndd = [] # diff
hvr = []; nlr = []; ndr = [] # ratio
for s in range(5000):
    np.random.seed(111)
    a = random.choice(df_K.index)
    np.random.seed(999)
    b = random.choice(df_K.index)
    x = df_K['Head-Volume'][a]; y = df_K['Head-Volume'][b]
    xn = df_K['Neck-Length'][a]; yn = df_K['Neck-Length'][b]
    xm = df_K['Neck-Radius'][a]*2; ym = df_K['Neck-Radius'][b]*2
    hvd.append(np.abs(x-y))
    nld.append(np.abs(xn-yn))
    ndd.append(np.abs(xm-ym))
    # ratio
    hvr.append(np.max([x,y])/np.min([x,y]))
    nlr.append(np.max([xn,yn])/np.min([xn,yn]))
    ndr.append(np.max([xm,ym])/np.min([xm,ym]))
    

### diff

fig = plt.figure(figsize=(11,7.5))
ax4 = fig.add_subplot(2, 3, 1)
sp = np.min(hvd); ep = np.max(hvd) # start/end point
ml = [i/len(hvd) for i in range(len(hvd))]
ml.append(1)
ax4.plot(np.append(np.sort(hvd), ep) , ml, 'b', label='Random pairs') # pairs
ol = [i/len(HV_dif) for i in range(len(HV_dif))]
ol.insert(0,0); ol.append(1)
ax4.plot(np.insert(np.append(np.sort(HV_dif),ep), 0, sp) , ol, 'r', label='Dual connection') # pairs
ax4.set_xlabel(r'$\Delta$ Head volume ($\mu m^{3}$)', fontsize=16)
ax4.set_ylabel('Cumulative probability', fontsize=16)
ax4.tick_params(axis="x", labelsize=16)
ax4.tick_params(axis="y", labelsize=16)
ax4.set_xscale('log')
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.annotate('A', xy=(-0.28, 1), xycoords='axes fraction', fontsize=16)

ax4 = fig.add_subplot(2, 3, 2)
sp = np.min(nld); ep = np.max(nld) # start/end point
ml = [i/len(nld) for i in range(len(nld))]
ml.append(1)
ax4.plot(np.append(np.sort(nld), ep) , ml, 'b')
ol = [i/len(NL_dif) for i in range(len(NL_dif))]
ol.insert(0,0)
ol.append(1)
ax4.plot(np.insert(np.append(np.sort(NL_dif), ep), 0, sp) , ol, 'r')
ax4.set_xlabel(r'$\Delta$ Neck length ($\mu m$)', fontsize=16)
ax4.tick_params(axis="x", labelsize=16)
ax4.tick_params(axis="y", labelsize=16)
ax4.set_xscale('log')
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.annotate('B', xy=(-0.23, 1), xycoords='axes fraction', fontsize=16)

ax4 = fig.add_subplot(2, 3, 3)
sp = np.min(ndd); ep = np.max(ndd) # start/end point
ml = [i/len(ndd) for i in range(len(ndd))]
ml.append(1)
ax4.plot(np.append(np.sort(ndd), ep) , ml, 'b')
ol = [i/len(ND_dif) for i in range(len(ND_dif))]
ol.insert(0,0); ol.append(1)
ax4.plot(np.insert(np.append(np.sort(ND_dif),ep), 0, sp) , ol, 'r')
ax4.set_xlabel(r'$\Delta$ Neck diameter ($\mu m$)', fontsize=16)
#ax4.set_ylabel('Cumulative probability', fontsize=16)
ax4.tick_params(axis="x", labelsize=16)
ax4.tick_params(axis="y", labelsize=16)
ax4.set_xscale('log')
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.annotate('C', xy=(-0.23, 1), xycoords='axes fraction', fontsize=16)



### ratio
ax4 = fig.add_subplot(2, 3, 4)
sp = np.min(hvr); ep = np.max(hvr) # start/end point
ml = [i/len(hvr) for i in range(len(hvr))]
ml.append(1)
ax4.plot(np.append(np.sort(hvr), ep) , ml, 'b', label='Random') # pairs
ol = [i/len(HV_rat) for i in range(len(HV_rat))]
ol.insert(0,0); ol.append(1)
ax4.plot(np.insert(np.append(np.sort(HV_rat),ep), 0, sp) , ol, 'r', label='Dual connection') # pairs
ax4.set_xlabel('Head volumes ratio', fontsize=16)
ax4.set_ylabel('Cumulative probability', fontsize=16)
ax4.tick_params(axis="x", labelsize=16)
ax4.tick_params(axis="y", labelsize=16)
ax4.set_xscale('log')
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.set_xlim([0.74, 30])
ax4.annotate('D', xy=(-0.28, 1), xycoords='axes fraction', fontsize=16)

ax4 = fig.add_subplot(2, 3, 5)
sp = np.min(nlr); ep = np.max(nlr) # start/end point
ml = [i/len(nlr) for i in range(len(nlr))]
ml.append(1)
ax4.plot(np.append(np.sort(nlr), ep) , ml, 'b')
ol = [i/len(NL_rat) for i in range(len(NL_rat))]
ol.insert(0,0)
ol.append(1)
ax4.plot(np.insert(np.append(np.sort(NL_rat), ep), 0, sp) , ol, 'r')
ax4.set_xlabel('Neck lengths ratio', fontsize=16)
ax4.tick_params(axis="x", labelsize=16)
ax4.tick_params(axis="y", labelsize=16)
ax4.set_xscale('log')
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.annotate('E', xy=(-0.23, 1), xycoords='axes fraction', fontsize=16)

ax6 = fig.add_subplot(2, 3, 6)
sp = np.min(ndr); ep = 10 # np.max(ndr) # start/end point
ml = [i/len(ndr) for i in range(len(ndr))]
ml.append(1)
ax6.plot(np.append(np.sort(ndr), ep) , ml, 'b', label='Random pairs')
ol = [i/len(ND_rat) for i in range(len(ND_rat))]
ol.insert(0,0); ol.append(1)
ax6.plot(np.insert(np.append(np.sort(ND_rat),ep), 0, sp) , ol, 'r', label='Dual connection')
ax6.set_xlabel('Neck diameters ratio', fontsize=16)
ax6.set_xscale('log')
ax6.tick_params(axis="x", labelsize=16)
ax6.tick_params(axis="y", labelsize=16)
ax6.spines['right'].set_visible(False)
ax6.spines['top'].set_visible(False)
ax6.annotate('F', xy=(-0.23, 1), xycoords='axes fraction', fontsize=16)
ax6.set_xlim([0.91, 10])
ax6.legend(fontsize=14)
plt.tight_layout()
fileName3 = ("C:/Data_Spines/Kasthuri_spines/figures_paper2/Dual.pdf" )
#fig.savefig(fileName3)
plt.show()




# difference
ks_2samp(hvr, HV_dif) # pvalue=0.6475
ks_2samp(nlr, NL_dif) # pvalue=0.7694
ks_2samp(ndr, ND_dif) # pvalue=0.06709

# ratio
ks_2samp(hvr, HV_dif) # pvalue=0.0034
ks_2samp(nlr, NL_dif) # pvalue=0.5915
ks_2samp(ndr, ND_dif) # pvalue=0.1046


