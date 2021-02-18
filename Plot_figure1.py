import os.path
from math import sqrt
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import seaborn as sns
from sklearn.mixture import GaussianMixture
from numpy import *
from fnc import *
import math

#files_list = []
#path = 'C:\\Data_Spines\\Kasthuri_spines'
#for root, subFolders , files in os.walk(path):
#  for f in files:
#    if f.endswith('.off'):
#      files_list.append('%s\\%s' % (root , f))

#files_list.sort(key=natural_keys)

#path = files_list[3649:3650][0]
path = 'C:\\Data_Spines\\Kasthuri_spines\\off_files\\Kasthuri__4643_Spines.OD_45_Spine_2_-.off'

txt_file_name = ("C:/Data_Spines/Kasthuri_spines/Kasthuri_seg_SDF_skl/%s.txt" % (path.split("\\")[-1].split(".o")[-2]))
print(txt_file_name)
fileName = txt_file_name.split("/")[-1].split(".t")[0]
cgal_file_name = ("C:/Data_Spines/Kasthuri_spines/skeleton_files/skel_%s.txt" % (path.split("\\")[-1].split(".o")[-2]))
cgal_file_name2 = ("C:/Data_Spines/Kasthuri_spines/skeleton_files/correspondance_%s.txt" % (path.split("\\")[-1].split(".o")[-2]))
if ((os.stat(cgal_file_name).st_size == 0) | (os.stat(cgal_file_name2).st_size == 0) | (os.stat(txt_file_name).st_size == 0)):
    print('continue')
lbl = path.split("\\")[-1].split(".o")[0]
for line in open(txt_file_name, 'r'):
    C = line.split(" ")
C = np.array(C)[:-1].astype(float)
#
if (np.count_nonzero(C == 0)<17):
    C = C - 1
if (np.count_nonzero(C == 0)<17):
    C = C - 1
C = np.array([0 if x < 1 else x for x in C])
uni = len(np.unique(C)) 

vert , nrml_vert , faces = load_OFF(path) 
vertices = np.array(np.float64(vert))
##
vertices = np.round(vertices,3)
##
faces = np.asarray(np.float64(faces).astype(int))
tris = vertices[faces]
##
## finding neighbors:
neighbors = find_neighbors(faces)
C = np.array([0 if x < 1 else 1 for x in C])
C = fltr2(C,neighbors)
# faces normals:
n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
n = normalize_v3(n)
n = np.nan_to_num(n)
#
## The Center Line
CL = []
for line in open(cgal_file_name, 'r'):
    CL.append( line.split(" ")[1:] )

for c in range(len(CL)):
    CL[c] = np.array(CL[c]).astype(float)
    CL[c] = CL[c].reshape(int(len(CL[c])/3),3)

CL = CL[np.argmax([len(l) for l in CL])] # take only the longer center line
# first edge point
rayDirection = CL[-2] - CL[-1]
if ((rayDirection==[0,0,0]).all()):
    rayDirection = CL[-3] - CL[-2]
rayPoint = CL[-1]
SD = []
edg2 = []
for ff in range(len(tris)):
    planeNormal = -n[ff]
    planePoint = tris[ff][0]
    Psi = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
    # is inside?
    [x1, y1, z1], [x2, y2, z2], [x3, y3, z3] = tris[ff]
    K = isInside(x1, y1, z1, x2, y2, z2, x3, y3, z3, Psi[0], Psi[1], Psi[2])
    if (K == True):
        SD.append(distance3d(CL[-1][0], CL[-1][1], CL[-1][2], Psi[0], Psi[1], Psi[2]))
        edg2.append([Psi[0], Psi[1], Psi[2]])

edg2d = [] # the direction of the continous center line:
for i in range(len(edg2)):
    if (np.sign(rayDirection)==np.sign(CL[-1]-edg2[i])).all():
        edg2d.append(edg2[i])
    
G = []
for i in range(len(edg2d)):
    G.append(distance3d(edg2d[i][0], edg2d[i][1], edg2d[i][2], CL[-1][0], CL[-1][1], CL[-1][2]))
    
if np.size(G)==0:
    edg3 = edg2[0]
else:
    edg3 = edg2d[np.argmin(G)]
# second edge point (neck - blue)
rayDirection = CL[1] - CL[0]
if ((rayDirection==[0,0,0]).all()):
    rayDirection = CL[2] - CL[1]
rayPoint = CL[0]
SD = []
edg2 = []
for ff in range(len(tris)):
    planeNormal = -n[ff]
    planePoint = tris[ff][0]
    Psi = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
    # is inside?
    [x1, y1, z1], [x2, y2, z2], [x3, y3, z3] = tris[ff]
    K = isInside(x1, y1, z1, x2, y2, z2, x3, y3, z3, Psi[0], Psi[1], Psi[2])
    if (K == True):
        SD.append(distance3d(CL[0][0], CL[0][1], CL[0][2], Psi[0], Psi[1], Psi[2]))
        edg2.append([Psi[0], Psi[1], Psi[2]])

edg2d = [] # the direction of the continous center line:
for i in range(len(edg2)):
    if (np.sign(rayDirection)==np.sign(CL[0]-edg2[i])).all():
        edg2d.append(edg2[i])
            
G = []
for i in range(len(edg2d)):
    G.append(distance3d(edg2d[i][0], edg2d[i][1], edg2d[i][2], CL[0][0], CL[0][1], CL[0][2]))
    
if np.size(G)==0:
    edg4 = edg2[0]
else:
    edg4 = edg2d[np.argmin(G)]
#
## Mean associated to a skeleton vertex of the edge dots.
MA = []
for line in open(cgal_file_name2, 'r'):
    MA.append(line.split(" ")[1:])

MA = np.array(MA).astype(float)
MA = MA.reshape(int(len(MA)*2),3)
##
MA = np.round(MA,3)
CL = np.round(CL,3)
##
C_vert = []
for v in vertices:
    BB = np.unique(np.where((v[0] == tris[:,:,0]) & (v[1] == tris[:,:,1]) &(v[2] == tris[:,:,2]) )[0])
    C_vert.append( sum(C[BB])/len(C[BB]) )

All_vert = np.zeros(len(CL))
Head_vert = np.zeros(len(CL))
vert_pos_in_MA = []
vv = 0
for v in vertices:
    B = np.where((v[0] == MA[:,0]) & (v[1] == MA[:,1]) &(v[2] == MA[:,2]) ) #[0][0]
    if np.size(B)==0:
        B = np.where(((v[0] == MA[:,0])|(v[0] == MA[:,0]-0.001)|(v[0] == MA[:,0]+0.001)) &  ((v[1] == MA[:,1])|(v[1] == MA[:,1]-0.001)|(v[1] == MA[:,1]+0.001)) & ((v[2] == MA[:,2])|(v[2] == MA[:,2]-0.001)|(v[2] == MA[:,2]+0.001)))[0][0]
    else:
        B = B[0][0]
    pos_CL = np.where((CL[:,0] == MA[B - 1][0]) & (CL[:,1] == MA[B - 1][1]) & (CL[:,2] == MA[B - 1][2]) )[0] #[0] # pos in CL that vert belong to
    if (len(pos_CL)>0):
        pos_CL = pos_CL[0]
    All_vert[pos_CL] = All_vert[pos_CL] + 1
    Head_vert[pos_CL] = Head_vert[pos_CL] + C_vert[vv]
    vv = vv + 1

All_vert[All_vert==0] = 1 # prevent divided by zero
Head_vert = Head_vert/All_vert
Head_vert = np.array(Head_vert)
Head_vert[Head_vert<0.5] = 0
Head_vert[Head_vert>0] = 1
# low pass filter:
B = np.zeros(len(Head_vert))
for i in range(1,len(Head_vert)-1):
    B[i] = np.mean([Head_vert[i-1], Head_vert[i], Head_vert[i+1]])

B[0] = B[1]; B[-1] = B[-2]
B[B<0.5] = 0; B[B>0] = 1
Head_vert = B
#
tris_head = tris[C>0]
# finding neighbors, only for the head:
neighbrs = []
border_edge = []
border_edge_idx = []
for i in range(len(tris_head)):
    nn = []
    mm = []
    mm_idx = []
    for j in range(len(tris_head)):
        aa = tris_head[i].tolist()
        b = tris_head[j].tolist()
        if len([x for x in aa if x in b])==2:  #  intersection
            nn.append(j)
            mm.append([x for x in aa if x not in b][0])
            mm_idx.append(aa.index([x for x in aa if x not in b][0]))
    neighbrs.append(nn)
    border_edge.append(mm)
    border_edge_idx.append(mm_idx)
# The number of neighbors of each face:
num_nei = np.array([len(neighbrs[i]) for i in range(len(neighbrs))])
tris_border = tris_head[num_nei<3]
border_edge = np.array(border_edge)[num_nei<3]
border_edge_idx = np.array(border_edge_idx)[num_nei<3]
#
hole_cntr = np.mean(np.mean(tris_border,axis=0),axis=0) + 0.0000001
tris_head_full = tris_head
for i in range(len(border_edge)):
    new_face = [[]]*3
    k = 0
    for p in border_edge_idx[i][::-1]:
        new_face[p] = border_edge[i][k]
        k = k + 1
    new_face[3-border_edge_idx[i][0] - border_edge_idx[i][1]] = hole_cntr.tolist()
    tris_head_full = np.concatenate((tris_head_full , [new_face]))
# Neck Radius - NEW:
mean_rad = []
for i in range(1,len(CL)-1):
    rad = []
    if ((Head_vert[i]==0)&(Head_vert[i+1]==0)):    
        for m in range(0,len(MA),2):
            if ((CL[i,0] == MA[m][0]) & (CL[i,1] == MA[m][1]) & (CL[i,2] == MA[m][2]) ):
                rad.append(np.array(MA[m+1]))
                if (i==40):
                    print(i,m)
        rad = [x for x in rad if str(x) != 'nan']
        mean_rad.append( np.mean( dist_line_point(p=CL[i], q=CL[i+1], rs=rad) ) )
mean_rad = [x for x in mean_rad if str(x) != 'nan']





# Gray
fig = plt.figure(figsize=(2.5,5))
ax = fig.gca(projection='3d')
ax.add_collection3d(Poly3DCollection(tris, facecolors='gray', edgecolor='k', linewidths=0.1, alpha=1.0))
scl = np.max( [np.max(np.transpose(vertices)[0])-np.min(np.transpose(vertices)[0]) , np.max(np.transpose(vertices)[1])-np.min(np.transpose(vertices)[1]) , np.max(np.transpose(vertices)[2])-np.min(np.transpose(vertices)[2])] )/2
ax.set_xlim(np.mean(np.transpose(vertices)[0])-scl, np.mean(np.transpose(vertices)[0])+scl)
ax.set_ylim(np.mean(np.transpose(vertices)[1])-scl, np.mean(np.transpose(vertices)[1])+scl)
ax.set_zlim(np.mean(np.transpose(vertices)[2])-scl, np.mean(np.transpose(vertices)[2])+scl)
plt.axis('off')
ax.view_init(elev=-40, azim=-75)
# Scale bar: 100nm
SP = [48.82, 28.8, -37.4] # starting point
ax.plot3D([SP[0], SP[0]+.1] , [SP[1], SP[1]] , [SP[2], SP[2]] , '-', color='k', linewidth=2)
ax.plot3D([SP[0], SP[0]] , [SP[1], SP[1]-.1] , [SP[2], SP[2]] , '-', color='k', linewidth=2)
ax.plot3D([SP[0], SP[0]] , [SP[1], SP[1]] , [SP[2], SP[2]+0.1] , '-', color='k', linewidth=2)
#fig.tight_layout()
fig.subplots_adjust(left=-0.5, right=1.5, bottom=0.0, top=1.0) #
ax.annotate('A', xy=(0.26, 0.93), xycoords='axes fraction', fontsize=16)
fileName3 = ("C:/Data_Spines/Kasthuri_spines/figures_paper/%s_gray2.pdf" % (fileName))
#fig.savefig(fileName3)
print(fileName)
plt.show()



cntrs = tris.mean(axis=1)
SDF = []
PS = []
oo = 100
for rays in range(15):  # 30
    SD = []
    rayDirection = randSphericalCap(coneAngleDegree=60, coneDir=-n[oo], N=1, seed=rays)[0]
    rayPoint = cntrs[oo]
    for ff in range(len(tris)):
        planeNormal = -n[ff]
        planePoint = cntrs[ff]
        Psi = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
        # is inside?
        [x1, y1, z1], [x2, y2, z2], [x3, y3, z3] = tris[ff]
        K = isInside(x1, y1, z1, x2, y2, z2, x3, y3, z3, Psi[0], Psi[1], Psi[2])
        if (K == True) & (oo != ff):
            PS.append(Psi)
    
PS2 = []
for p in PS:
    if (np.sum(p)!=0):
        PS2.append(p)
# Red
fig = plt.figure(figsize=(2.5,5))
ax = fig.gca(projection='3d')
clr = ['b', 'g', 'r', 'y' ,'pink' , 'k']
ax.add_collection3d(Poly3DCollection(tris, facecolors='gray', edgecolor='k', linewidths=0.1, alpha=0.0)) # 0.2
X = np.array([np.transpose(CL)[0], np.transpose(CL)[1],np.transpose(CL)[2]])
for i in range(len(X[0])-1):
    ax.plot3D([X[0][i], X[0][i+1]], [X[1][i], X[1][i+1]], [X[2][i], X[2][i+1]],  color='r', linewidth=3)
i,m = 40 , 484
ax.plot3D([CL[i][0], MA[m+1][0]], [CL[i][1], MA[m+1][1]], [CL[i][2], MA[m+1][2]], color='orange' , linewidth=5) # 
for i in range(25):
    ax.plot3D([rayPoint[0], PS2[i][0]] , [rayPoint[1], PS2[i][1]] , [rayPoint[2], PS2[i][2]], color='purple', linewidth=2)
ax.set_xlim(np.mean(np.transpose(vertices)[0])-scl, np.mean(np.transpose(vertices)[0])+scl)
ax.set_ylim(np.mean(np.transpose(vertices)[1])-scl, np.mean(np.transpose(vertices)[1])+scl)
ax.set_zlim(np.mean(np.transpose(vertices)[2])-scl, np.mean(np.transpose(vertices)[2])+scl)
plt.axis('off')
ax.view_init(elev=-40, azim=-75)
fig.subplots_adjust(left=-0.5, right=1.5, bottom=0.0, top=1.0)
ax.annotate('B', xy=(0.26, 0.93), xycoords='axes fraction', fontsize=16)
fileName3 = ("C:/Data_Spines/Kasthuri_spines/figures_paper/%s_red.pdf" % (fileName))
#fig.savefig(fileName3)
print(fileName)
plt.show()



# Blue + Green
fig = plt.figure(figsize=(2.5,5))
ax = fig.gca(projection='3d')
clr = ['b', 'g', 'r', 'y' ,'pink' , 'k']
ax.add_collection3d(Poly3DCollection(tris, facecolors=np.array(clr)[C.astype(int)], edgecolor='k', linewidths=0.1, alpha=0.3))
X = np.array([np.transpose(CL)[0], np.transpose(CL)[1], np.transpose(CL)[2]])
for i in range(len(X[0])-1):
    ax.plot3D([X[0][i], X[0][i+1]], [X[1][i], X[1][i+1]], [X[2][i], X[2][i+1]],  color=np.array(clr)[Head_vert[i].astype(int)], linewidth=3)
ax.plot3D([edg3[0], CL[-1][0]], [edg3[1], CL[-1][1]], [edg3[2], CL[-1][2]], color=np.array(clr)[Head_vert[-1].astype(int)],linewidth=3) #, '.-'
ax.plot3D([edg4[0], CL[0][0]], [edg4[1], CL[0][1]], [edg4[2], CL[0][2]], color=np.array(clr)[Head_vert[0].astype(int)],linewidth=3)
ax.set_xlim(np.mean(np.transpose(vertices)[0])-scl, np.mean(np.transpose(vertices)[0])+scl)
ax.set_ylim(np.mean(np.transpose(vertices)[1])-scl, np.mean(np.transpose(vertices)[1])+scl)
ax.set_zlim(np.mean(np.transpose(vertices)[2])-scl, np.mean(np.transpose(vertices)[2])+scl)
plt.axis('off')
fileName3 = ("C:/Data_Spines/Kasthuri_spines/figures_paper/%s_blue.pdf" % (fileName))
ax.view_init(elev=-40, azim=-75)
fig.subplots_adjust(left=-0.5, right=1.5, bottom=0.0, top=1.0)
ax.annotate('C', xy=(0.26, 0.93), xycoords='axes fraction', fontsize=16)
#fig.savefig(fileName3)
print(fileName)
plt.show()




# Figure 1D
txt_file_name_SDF = ("C:/Data_Spines/Kasthuri_spines/Kasthuri_SDF_seg_values/%s.txt" % (path.split("\\")[-1].split(".o")[-2]))    
for line in open(txt_file_name_SDF, 'r'):
    SDF = line.split(" ")
SDF = np.array(SDF)[:-1].astype(float)
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
M = pd.DataFrame(np.transpose([SDF, rad_f ])) #
gmm = GaussianMixture(n_components=2,covariance_type='spherical') #
gmm.fit(M)
score2 = gmm.score(M)
pred = gmm.predict(M)
if (np.mean(SDF[pred==0]) > np.mean(SDF[pred==1])):
    pred = 1 - pred
    


##
sns_plot = sns.jointplot(x=SDF, y=rad_f, kind='kde', color='r',alpha=.5, height=5)
sns_plot.fig.text(0.05, 0.93, 'D' , fontsize = 16)
sns_plot.plot_joint(plt.scatter, c=np.array(clr)[pred], s=5, linewidth=.7, marker=".",alpha=1)
plt.xlabel('SDF', fontsize=16)
plt.ylabel('Skeleton radius', fontsize=16)
plt.setp(sns_plot.ax_marg_y.patches, color="r")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
fileName3 = ("C:/Data_Spines/Kasthuri_spines/figures_paper/Scatter.pdf" )
#sns_plot.savefig(fileName3)
plt.show()




