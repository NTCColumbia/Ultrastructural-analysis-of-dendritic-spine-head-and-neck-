import os.path
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import math
import pandas as pd
import random
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_OBJ(path,smoothing=False):
    f = open(path, 'r')
    vert = []
    nrml_vert = []
    faces = []
    for line in f.readlines():
        if line[0:2] == 'v ':
            vert.append(line.strip().split()[1:])
        if line[0:2] == 'vn':
            nrml_vert.append(line.strip().split()[1:])
        if line[0] == 'f':
            ll = [line.strip().split()[i].split('//')[0] for i in range(4)]
            faces.append([int(x) - 1 for x in ll[1:]])
    f.close()
    if smoothing==True:
        ######### finding verices neighbors:
        vert_nei= [set()]*len(vert)
        for i in range(len(faces)):
            for j in range(3):
                vert_nei[faces[i][j]] =  vert_nei[faces[i][j]] | {faces[i][(j+1)%3],faces[i][(j+2)%3] }
        # Laplacian smoothing:
        vert = np.asarray(np.float64(vert))
        new_vert= []
        for i in range(len(vert)):
            new_vert.append( np.mean(np.array(vert)[list(vert_nei[i])],axis=0) )
        vert = np.around(new_vert, decimals=3)
    return vert , nrml_vert , faces


def load_OFF(path):
    f = open(path, 'r')
    vert = []
    nrml_vert = []
    faces = []
    for line in f.readlines():
        if line[0:2] == '3 ':
            ll = [line.strip().split()[i].split('//')[0] for i in range(4)]
            faces.append([int(x) - 0 for x in ll[1:]])
        else:
            vert.append(line.strip().split()) # [1:]
    f.close()
    return vert[2:] , nrml_vert , faces


# https://math.stackexchange.com/questions/128991/how-to-calculate-the-area-of-a-3d-triangle
def heron(a, b, c):
    s = (a + b + c) / 2
    area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    return area


def distance3d(x1, y1, z1, x2, y2, z2):
    a = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
    d = a ** 0.5
    return d


def areatriangle3d(zz):  # x1, y1, z1, x2, y2, z2, x3, y3, z3):
    x1, y1, z1 = zz[0]
    x2, y2, z2 = zz[1]
    x3, y3, z3 = zz[2]
    a = distance3d(x1, y1, z1, x2, y2, z2)
    b = distance3d(x2, y2, z2, x3, y3, z3)
    c = distance3d(x3, y3, z3, x1, y1, z1)
    A = heron(a, b, c)
    # print("area of triangle is %r " % A)
    return A


#  Zhang and Chen 2001
def SignedVolumeOfTriangle(zz):
    x1, y1, z1 = zz[0]
    x2, y2, z2 = zz[1]
    x3, y3, z3 = zz[2]
    v321 = x3 * y2 * z1
    v231 = x2 * y3 * z1
    v312 = x3 * y1 * z2
    v132 = x1 * y3 * z2
    v213 = x2 * y1 * z3
    v123 = x1 * y2 * z3
    return (1.0 / 6.0) * (-v321 + v231 + v312 - v132 - v213 + v123)


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


# https://www.geeksforgeeks.org/check-whether-a-given-point-lies-inside-a-triangle-or-not/
def isInside(x1, y1, z1, x2, y2, z2, x3, y3, z3, x, y, z):
    # Calculate area of triangle ABC
    A = areatriangle3d([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
    # Calculate area of triangle PBC
    A1 = areatriangle3d([[x, y, z], [x2, y2, z2], [x3, y3, z3]])
    # Calculate area of triangle PAC
    A2 = areatriangle3d([[x1, y1, z1], [x, y, z], [x3, y3, z3]])
    # Calculate area of triangle PAB
    A3 = areatriangle3d([[x1, y1, z1], [x2, y2, z2], [x, y, z]])
    # Check if sum of A1, A2 and A3 is same as A
    ep = 0.0000000001  # sys.float_info.epsilon #
    if (A > (A1 + A2 + A3 - ep)) & (A < (A1 + A2 + A3 + ep)):
        return True
    else:
        return False


# https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Python
def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        return [0, 0, 0]
        # raise RuntimeError("no intersection or line is within plane")
    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi


def solve(m1, m2, std1, std2):
    a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
    b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
    c = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / (2 * std2 ** 2) - np.log(std2 / std1)
    return np.roots([a, b, c])


def gauss(x, mu, sigma, A):
    import numpy as np
    return A * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2) # (A/(sigma*np.sqrt(2*np.pi)))


def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2)


def plt_spine(tris, vertices, BC=None ,title=''):
    if BC is None:
        BC = np.zeros(len(tris))

    fig = plt.figure()
    ax = Axes3D(fig)
    clr = ['b', 'g', 'r', 'y' ,'pink' , 'k']
    ax.add_collection3d(Poly3DCollection(tris, facecolors=np.array(clr)[BC.astype(int)], edgecolor='k', linewidths=0.1, alpha=0.4))
    # ax.add_collection3d(Poly3DCollection(tris , facecolors='green', edgecolor='k', linewidths=0.2, alpha=0.4))
    # np.array(clr)[np.abs(np.sum([conv],axis=2)).astype(int).tolist()[0]]  #  for convex
    # ax.scatter(x, y, z, marker='.', s=1, c="black", alpha=0.5)
    ax.scatter(*np.transpose(vertices), marker='.', s=0.5, c="black", alpha=0.5)
    scl = np.max( [np.max(np.transpose(vertices)[0])-np.min(np.transpose(vertices)[0]) , np.max(np.transpose(vertices)[1])-np.min(np.transpose(vertices)[1]) , np.max(np.transpose(vertices)[2])-np.min(np.transpose(vertices)[2])] )/2
    ax.set_xlim(np.mean(np.transpose(vertices)[0])-scl, np.mean(np.transpose(vertices)[0])+scl)
    ax.set_ylim(np.mean(np.transpose(vertices)[1])-scl, np.mean(np.transpose(vertices)[1])+scl)
    ax.set_zlim(np.mean(np.transpose(vertices)[2])-scl, np.mean(np.transpose(vertices)[2])+scl)
    #ax.set_xlim(min(np.transpose(vertices)[0]), max(np.transpose(vertices)[0]))  # min(x), max(x))
    #ax.set_ylim(min(np.transpose(vertices)[1]), max(np.transpose(vertices)[1]))
    #ax.set_zlim(min(np.transpose(vertices)[2]), max(np.transpose(vertices)[2]))
    ax.view_init(elev=-150, azim=-50)  # 30 , -100
    #plt.title("{fle}".format(fle=path.split('/')[-1]))
    plt.title(title)
    plt.axis('off')
    return(fig)


def fltr(CC, neighbors, nei_of_nei):
    ## filters
    bb = np.array(neighbors)[neighbors].tolist()
    C = CC
    for t in range(10):
        # change if at least 3 neighbors
        C[np.sum(C[neighbors], axis=1) < 1] = 0
        C[np.sum(C[neighbors], axis=1) > 2] = 1
        # change if at least 2 neighbors
        C[np.sum(C[neighbors], axis=1) < 2] = 0
        C[np.sum(C[neighbors], axis=1) > 1] = 1
        for i in range(len(bb)):
            if (np.mean(C[nei_of_nei[i]]) > 0.5):
                C[i] = 1
            elif (np.mean(C[nei_of_nei[i]]) < 0.5):
                C[i] = 0
    # change if at least 2 neighbors
    C[np.sum(C[neighbors], axis=1) < 2] = 0
    C[np.sum(C[neighbors], axis=1) > 1] = 1
    return C


def fltr2(CC, neighbors):
    ## filters
    #bb = np.array(neighbors)[neighbors].tolist()
    C = np.array(CC)
    for t in range(2):
        # change if at least 3 neighbors
        C[np.sum(C[neighbors], axis=1) < 1] = 0
        C[np.sum(C[neighbors], axis=1) > 2] = 1
        # change if at least 2 neighbors
        C[np.sum(C[neighbors], axis=1) < 2] = 0
        C[np.sum(C[neighbors], axis=1) > 1] = 1
    return C


def find_neighbors_old(faces):
    # finding faces neighbors:
    nn = []
    neighbors = []
    for i in range(len(faces)):
        for j in range(len(faces)):
            if len(list(set(faces[i]) & set(faces[j]))) == 2:  # intersection
                nn.append(j)

        neighbors.append(nn)
        nn = []
    return neighbors


def find_neighbors(faces):
    # finding faces neighbors:
    neighbors = []
    for i in range(len(faces)):
        nn = np.where(faces[i][0] == faces)[0].tolist() + np.where(faces[i][1] == faces)[0].tolist() +np.where(faces[i][2] == faces)[0].tolist()
        neighbors.append(list(set([i for i in nn if nn.count(i)==2])))
    return neighbors


# finding vertices_neighbors
def find_vertices_neighbors(faces,len_vertices):
    vert_neighbors = [[] for x in range(len_vertices)]
    for v in faces:
        vert_neighbors[v[0]].extend(v[1:3])
        vert_neighbors[v[1]].extend([v[0], v[2]])
        vert_neighbors[v[2]].extend(v[0:2])

    for vn in range(len(vert_neighbors)):
        vert_neighbors[vn] = np.unique(vert_neighbors[vn])
    return vert_neighbors


def SDF_faces(tris,n):
    cntrs = tris.mean(axis=1)
    SDF = []
    for oo in range(len(tris)):
        SD = []
        rayDirection = -n[oo]
        rayPoint = cntrs[oo]
        for ff in range(len(tris)):
            planeNormal = -n[ff]
            planePoint = cntrs[ff]
            Psi = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
            # is inside?
            [x1, y1, z1], [x2, y2, z2], [x3, y3, z3] = tris[ff]
            K = isInside(x1, y1, z1, x2, y2, z2, x3, y3, z3, Psi[0], Psi[1], Psi[2])
            if (K == True) & (oo != ff):
                # print(distance3d(cntrs[0][0],cntrs[0][1],cntrs[0][2],Psi[0], Psi[1], Psi[2]))
                #SD.append(distance3d(cntrs[0][0], cntrs[0][1], cntrs[0][2], Psi[0], Psi[1], Psi[2]))
                SD.append(distance3d(cntrs[oo][0], cntrs[oo][1], cntrs[oo][2], Psi[0], Psi[1], Psi[2]))
        SD = list(filter(lambda a: a != 0, SD))
        if SD != []:
            SDF.append(np.min(SD)) 
        else:
            SDF.append(0.0)
    return np.array(SDF)


def SDF_vert(tris,vertices,nrml_vert,n):
    SDF = []
    for oo in range(len(vertices)):
        SD = []
        rayDirection = -nrml_vert[oo]
        rayPoint = vertices[oo]
        for ff in range(len(tris)):
            planeNormal = -n[ff]
            planePoint = tris[ff][0]  # cntrs[ff]
            Psi = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
            # is inside?
            [x1, y1, z1], [x2, y2, z2], [x3, y3, z3] = tris[ff]
            K = isInside(x1, y1, z1, x2, y2, z2, x3, y3, z3, Psi[0], Psi[1], Psi[2])
            if (K == True):  # & (oo != ff):
                SD.append(distance3d(vertices[oo][0], vertices[oo][1], vertices[oo][2], Psi[0], Psi[1], Psi[2]))
        SD = list(filter(lambda a: a != 0, SD))
        if SD != []:
            SDF.append(np.min(SD))
        else:
            SDF.append(0.0)
    return np.array(SDF)


def SDF_mistake(tris,vertices,nrml_vert,n):
    SDF = []
    for oo in range(len(vertices)):
        SD = []
        rayDirection = -nrml_vert[oo]
        rayPoint = vertices[oo]
        for ff in range(len(tris)):
            planeNormal = -n[ff]
            planePoint = tris[ff][0]  # cntrs[ff]
            Psi = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
            # is inside?
            [x1, y1, z1], [x2, y2, z2], [x3, y3, z3] = tris[ff]
            K = isInside(x1, y1, z1, x2, y2, z2, x3, y3, z3, Psi[0], Psi[1], Psi[2])
            if (K == True):  # & (oo != ff):
                SD.append(distance3d(min(vertices.T[0]), min(vertices.T[1]), min(vertices.T[2]), Psi[0], Psi[1], Psi[2]))
        SD = list(filter(lambda a: a != 0, SD))
        if SD != []:
            SDF.append(np.min(SD))
        else:
            SDF.append(0.0)
    return np.array(SDF)


def Spine_area_volume(tris):
    Area = 0.0 # Surface Area
    Volume = 0.0
    for tr in tris:
        Area = Area + areatriangle3d(tr)
        Volume = Volume + SignedVolumeOfTriangle(tr)
    return Area, Volume


def lngth(vertices):
    # the distance between the two most distance points
    dis = []
    loc = []
    for i in range(len(vertices)):
        for j in range(i+1,len(vertices)):
            dis.append(distance3d(*vertices[i],*vertices[j] ))
            loc.append([i,j])
    return np.max(dis) , loc[np.argmax(dis)][0] , loc[np.argmax(dis)][1]


import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


#https://stackoverflow.com/questions/50727961/shortest-distance-between-a-point-and-a-line-in-3-d-space
def dist_line_point(p, q, rs):
    x = p-q
    return np.linalg.norm(np.outer(np.dot(rs-q, x)/np.dot(x, x), x)+q-rs, axis=1)



def randSphericalCap(coneAngleDegree, coneDir, N, seed):
    coneAngle = coneAngleDegree * np.pi / 180
    # Generate points on the spherical cap around the north pole.
    # See https://math.stackexchange.com/a/205589/81266
    np.random.seed(seed)
    z = np.array( np.random.rand(1, N) * (1 - np.cos(coneAngle)) + np.cos(coneAngle) )
    np.random.seed(seed)
    phi = np.random.rand(1, N) * 2 * np.pi
    x = np.sqrt(1 - z**2) * np.cos(phi)
    y = np.sqrt(1 - z**2) * np.sin(phi)
    # Find the rotation axis `u` and rotation angle `rot`
    u = normc(np.cross([0,0,1], normc(coneDir)))
    rot = math.acos(np.dot(normc(coneDir), [0,0,1]))
    # Convert rotation axis and angle to 3x3 rotation matrix
    # See https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    R = np.cos(rot) * np.eye(3) + np.sin(rot) * crossMatrix(u[0], u[1], u[2]) + (1 - np.cos(rot)) * (u * u.T)
    # Rotate [x; y; z] from north pole to `coneDir`.
    r = np.dot(R , [x[0], y[0], z[0]])
    return r.T


def crossMatrix(x,y,z):
    M = np.array([[0,-z,y],[z, 0 , -x],[-y,x,0]])
    return M


def normc(x):
    # Normalize a numpy array of 3 component vectors shape=(n,3)
    lens = np.sqrt(x[0]**2 + x[1] ** 2 + x[2] ** 2)
    return x/lens