import numpy as np
from fnc import *

files_list = []
path = 'C:\\Data_Spines\\Kasthuri_spines'

for root, subFolders , files in os.walk(path):
  for f in files:
    if f.endswith('.obj'):
      files_list.append('%s\\%s' % (root , f))

files_list.sort(key=natural_keys)


#%% Creating batch file for SDF segmantation:

## Kasthuri
off_file_name = ("run_cpp_scripts_Kasthuri_SDF_skl.bat")
f= open(off_file_name,"w+")
for path in files_list:
    f.write("seg_SDF_skl off_files/{a}.off>Kasthuri_seg_SDF_skl/{a}.txt\n".format(a=path.split("\\")[-1].split(".o")[0] ))
f.close()

off_file_name = ("run_cpp_scripts_Kasthuri.bat")
f= open(off_file_name,"w+")
for path in files_list:
    f.write("SDF_seg off_files/{a}.off>Kasthuri_seg/{a}.txt\n".format(a=path.split("\\")[-1].split(".o")[0] ))
f.close()

off_file_name = ("run_cpp_scripts_Kasthuri_SDF_values.bat")
f= open(off_file_name,"w+")
for path in files_list:
    f.write("Project2 data_Kasthuri/{a}.off>Kasthuri_SDF_seg_values/{a}.txt\n".format(a=path.split("\\")[-1].split(".o")[0] ))
f.close()



