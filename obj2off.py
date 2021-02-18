import numpy as np
from fnc import *

files_list = []
path = 'C:\\Data_Spines\\Kasthuri_spines'

for root, subFolders , files in os.walk(path):
  for f in files:
    if f.endswith('.obj'):
      files_list.append('%s\\%s' % (root , f))

files_list.sort(key=natural_keys)
smooth = False # True 

for path in files_list:
    vert , nrml_vert , faces = load_OBJ(path,smoothing=smooth)
    vert = np.array(np.float64(vert))
    faces = np.asarray(np.float64(faces).astype(int))
    off_file_name = ("C:/Data_Spines/Kasthuri_spines/off_files/%s.off" % (path.split("\\")[-1].split(".o")[-2]))
    f= open(off_file_name,"w+")
    f.write("COFF\n")
    f.write("{a} {b} 0\n".format(a=len(vert) , b=len(faces)))
    for i in range(len(vert)):
        f.write("{a} {b} {c}\n".format(a=vert[i][0] , b= vert[i][1] , c= vert[i][2]))
    for i in range(len(faces)):
        f.write("3 {a} {b} {c}\n".format(a=faces[i][0] , b=faces[i][1] , c=faces[i][2]))
    f.close()

