import os.path
from math import sqrt
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import seaborn as sns
from numpy import *
from fnc import *

files_list = []
path = 'C:\\Data_Spines\\Kasthuri_spines'
for root, subFolders , files in os.walk(path):
  for f in files:
    if f.endswith('.off'):
      files_list.append('%s\\%s' % (root , f))

files_list.sort(key=natural_keys)

for path in files_list:
    txt_file_name = ("C:/Data_Spines/Kasthuri_spines/Kasthuri_seg_SDF_skl/%s.txt" % (path.split("\\")[-1].split(".o")[-2]))
    print(txt_file_name)
    fileName = txt_file_name.split("/")[-1].split(".t")[0]
    cgal_file_name = ("C:/Data_Spines/Kasthuri_spines/skeleton_files/skel_%s.txt" % (path.split("\\")[-1].split(".o")[-2]))
    cgal_file_name2 = ("C:/Data_Spines/Kasthuri_spines/skeleton_files/correspondance_%s.txt" % (path.split("\\")[-1].split(".o")[-2]))
    if ((os.stat(cgal_file_name).st_size == 0) | (os.stat(cgal_file_name2).st_size == 0) | (os.stat(txt_file_name).st_size == 0)):
        continue
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
    if (len(np.unique(C)) !=2 ): # > 3
        continue
    ## only for figure:
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
    if ((len(np.unique(C)) ==3) & (len(np.unique(C[neighbors][C==0])) > 2)): # discard twins
        continue
    C = np.array([0 if x < 1 else 1 for x in C])
    C = fltr2(C,neighbors)
    # faces normals:
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    n = normalize_v3(n)
    n = np.nan_to_num(n)
    #
    a,v = Spine_area_volume(tris)
    Area = a
    Volume = v
    ## Length:
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
    # Calculating the spine length
    spine_length = 0
    for i in range(len(CL)-1):
        spine_length = spine_length + distance3d(CL[i][0], CL[i][1], CL[i][2], CL[i+1][0], CL[i+1][1], CL[i+1][2])

    spine_length = spine_length + distance3d(CL[-1][0], CL[-1][1], CL[-1][2], edg3[0], edg3[1], edg3[2])
    spine_length = spine_length + distance3d(CL[0][0], CL[0][1], CL[0][2], edg4[0], edg4[1], edg4[2])
    SL = spine_length

    # Calculating the neck length
    neck_length = 0
    for i in range(len(CL)-1):
        neck_length = neck_length + (distance3d(CL[i][0], CL[i][1], CL[i][2], CL[i+1][0], CL[i+1][1], CL[i+1][2]) * (1-Head_vert[i]) )

    neck_length = neck_length + (distance3d(CL[-1][0], CL[-1][1], CL[-1][2], edg3[0], edg3[1], edg3[2]) * (1-Head_vert[-1]) )
    neck_length = neck_length + (distance3d(CL[0][0], CL[0][1], CL[0][2], edg4[0], edg4[1], edg4[2]) * (1-Head_vert[0]) )
    NL = neck_length
    #
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
    a,v = Spine_area_volume(tris_head_full)
    Area_Head = a
    Volume_Head = v
    # Neck Radius - NEW:
    mean_rad = []
    for i in range(1,len(CL)-1):
        rad = []
        if ((Head_vert[i]==0)&(Head_vert[i+1]==0)):    
            for m in range(0,len(MA),2):
                if ((CL[i,0] == MA[m][0]) & (CL[i,1] == MA[m][1]) & (CL[i,2] == MA[m][2]) ):
                    rad.append(np.array(MA[m+1]))
            rad = [x for x in rad if str(x) != 'nan']
            if (len(rad)>0):
                mean_rad.append( np.mean( dist_line_point(p=CL[i], q=CL[i+1], rs=rad) ) )
    mean_rad = [x for x in mean_rad if str(x) != 'nan']
    Neck_rad = np.nan_to_num(np.mean(mean_rad))
    ###
    myfile = open('C:/Data_Spines/Kasthuri_spines/Kasthuri_data_new10.txt', 'a')
    myfile.write("%s %s %s %s %s %s %s %s\n" % (lbl, Volume, Area, Volume_Head, Area_Head, SL, NL, Neck_rad))
    myfile.close()
    ###
    fig = plt.figure(figsize=(5,5))
    ax = Axes3D(fig)
    clr = ['b', 'g', 'r', 'y' ,'pink' , 'k']
    ax.add_collection3d(Poly3DCollection(tris, facecolors=np.array(clr)[C.astype(int)], edgecolor='k', linewidths=0.1, alpha=0.2))
    ax.scatter(np.transpose(CL)[0], np.transpose(CL)[1],np.transpose(CL)[2],  color=np.array(clr)[Head_vert.astype(int)],s=20)
    ax.scatter(edg3[0], edg3[1], edg3[2], marker='.', s=20, c=np.array(clr)[Head_vert[-1].astype(int)], alpha=1)
    ax.scatter(edg4[0], edg4[1], edg4[2], marker='.', s=20, c=np.array(clr)[Head_vert[0].astype(int)], alpha=1)
    ax.plot3D([edg3[0], CL[-1][0]], [edg3[1], CL[-1][1]], [edg3[2], CL[-1][2]], '.-', color=np.array(clr)[Head_vert[-1].astype(int)],linewidth=2)
    ax.plot3D([edg4[0], CL[0][0]], [edg4[1], CL[0][1]], [edg4[2], CL[0][2]], '.-', color=np.array(clr)[Head_vert[0].astype(int)],linewidth=2)
    scl = 1.25
    # Scale bar: 
    ax.set_xlim(np.mean(np.transpose(vertices)[0])-scl, np.mean(np.transpose(vertices)[0])+scl)
    ax.set_ylim(np.mean(np.transpose(vertices)[1])-scl, np.mean(np.transpose(vertices)[1])+scl)
    ax.set_zlim(np.mean(np.transpose(vertices)[2])-scl, np.mean(np.transpose(vertices)[2])+scl)
    plt.title(fileName)
    plt.axis('off')
    fileName2 = ("C:/Data_Spines/Kasthuri_spines/figures/%s.png" % (fileName))
    z = 10 * 0
    ax.view_init(elev=-150 + z, azim=-50 + z)
    #fig.savefig(fileName2, dpi=300)
    plt.show()
    #plt.close(fig)



