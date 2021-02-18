import os.path
import matplotlib.pyplot as plt
import numpy as np
from fnc import *
import seaborn as sns
from unidip import UniDip
import unidip.dip as dip
from matplotlib.ticker import MaxNLocator

files_list = []
path = 'C:\\Data_Spines\\Kasthuri_spines'

for root, subFolders , files in os.walk(path):
  for f in files:
    if f.endswith('.off'):
      files_list.append('%s\\%s' % (root , f))

files_list.sort(key=natural_keys)


for path in files_list[1413:]: # not work: 1218 (Kasthuri__1253), 1412 (Kasthuri__1459)
    txt_file_name = ("C:/Data_Spines/Kasthuri_spines/Kasthuri_seg_SDF_skl/%s.txt" % (path.split("\\")[-1].split(".o")[-2]))
    txt_file_name_SDF = ("C:/Data_Spines/Kasthuri_spines/Kasthuri_SDF_seg_values/%s.txt" % (path.split("\\")[-1].split(".o")[-2]))
    cgal_file_name2 = ("C:/Data_Spines/Kasthuri_spines/skeleton_files/correspondance_%s.txt" % (path.split("\\")[-1].split(".o")[-2]))
    fileName = txt_file_name.split("/")[-1].split(".t")[0]
    print(txt_file_name)
    if ((os.stat(txt_file_name).st_size == 0) | (os.stat(txt_file_name_SDF).st_size == 0) | (os.stat(cgal_file_name2).st_size == 0)):
        continue
    lbl = path.split("\\")[-1].split(".o")[0]
    for line in open(txt_file_name_SDF, 'r'):
        SDF = line.split(" ")
    SDF = np.array(SDF)[:-1].astype(float)
    # 
    vert , nrml_vert , faces = load_OFF(path) 
    vertices = np.array(np.float64(vert))
    #
    ## Mean associated to a skeleton vertex of the edge dots.
    MA = []
    for line in open(cgal_file_name2, 'r'):
        MA.append(line.split(" ")[1:])
    MA = np.array(MA).astype(float)
    MA = MA.reshape(int(len(MA)*2),3)
    ##
    MA = np.round(MA,3)
    MA_P = np.round(MA + 0.001 , 3)
    MA_M = np.round(MA - 0.001 , 3)
    vertices_round = np.round(vertices,3)
    vv = 0
    rad = []
    for vr in vertices_round:
        B = np.where((vr[0] == MA[:,0]) & (vr[1] == MA[:,1]) &(vr[2] == MA[:,2]) )
        if np.size(B)==0:
            B = np.where(((vr[0] == MA[:,0])|(vr[0] == MA_M[:,0])|(vr[0] == MA_P[:,0])) &  ((vr[1] == MA[:,1])|(vr[1] == MA_M[:,1])|(vr[1] == MA_P[:,1])) & ((vr[2] == MA[:,2])|(vr[2] == MA_M[:,2])|(vr[2] == MA_P[:,2])))[0][0]
        else:
            B = B[0][0]
        rad.append(distance3d(vr[0], vr[1], vr[2], MA[B-1][0], MA[B-1][1], MA[B-1][2]))
    ##
    rad = np.array(rad)
    rad_f = [] # radius for each face
    for ff in faces:
        rad_f.append( np.mean(rad[ff]) )
    #
    ##Normalized Data
    rad_f = (rad_f-min(rad_f))/(max(rad_f)-min(rad_f))
    ##
    sim = np.transpose([SDF, rad_f])
    # Moving the center to zero
    sim[:,0] = sim[:,0]  - np.min(sim[:,0])- (np.max(sim[:,0])-np.min(sim[:,0]))/2 #
    sim[:,1] = sim[:,1]  - np.min(sim[:,1])- (np.max(sim[:,1])-np.min(sim[:,1]))/2 #
    #
    dip_ang = [] 
    for ii in range(18):
        # Rotating matrix
        theta = np.radians(ii*10)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        rot = np.dot(sim, R)
        data = np.msort(rot[:,0])
        aa, p_dip, c = dip.diptst(data)
        dip_ang.append(p_dip)
        if (p_dip < 0.001):
            break
    ##
    if (ii<17):
        data = np.msort(rad_f)
        _, p_dip_rad, c = dip.diptst(data)
    else:
        p_dip_rad = dip_ang[-1]

    myfile = open('C:/Data_Spines/Kasthuri_spines/Kasthuri-Unimodal_tests_rotate.txt', 'a')
    myfile.write("%s %s %s %s\n" % (lbl, dip_ang[0], p_dip_rad, np.min(dip_ang)))
    myfile.close()







# Analysis:
df_dip = pd.read_csv('C:/Data_Spines/Kasthuri_spines/Kasthuri-Unimodal_tests_rotate.txt', sep=" ", header=None) 

ind = []
for i in range(len(df_dip)):
    ind.append(df_dip[0][i])
df_dip.index = ind
df_dip = df_dip.drop([0], axis=1)
df_dip.columns = ["p_dip_SDF", "p_dip_rad", "p_dip_rot"  ]
#
len(df_dip['p_dip_rad'][df_dip['p_dip_rad']<0.05]) # 3147
len(df_dip['p_dip_rad'][df_dip['p_dip_rad']<0.05]) / len(df_dip['p_dip_rad']) # 74.875%

len(df_dip['p_dip_SDF'][df_dip['p_dip_SDF']<0.05]) # 3717
len(df_dip['p_dip_SDF'][df_dip['p_dip_SDF']<0.05]) / len(df_dip['p_dip_SDF']) # 88.44%

len(df_dip['p_dip_rot'][df_dip['p_dip_rot']<0.05]) # 3998
len(df_dip['p_dip_rot'][df_dip['p_dip_rot']<0.05]) / len(df_dip['p_dip_rot']) # 95.12%


