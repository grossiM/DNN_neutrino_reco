import os
import sys
import configparser

import fnmatch
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import argparse
mypath = '/Users/it058990/Desktop/MicheleG/PhD/VBS_COST_Group/GITBicocca/DNN_storage/evaluation/ful_lep_VBS_WWmuvev_phantom_1_6_nob/cos_theta_sel4/all_hist_toplot/'
from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

long_hist = {}
trans_hist = {}

for a in onlyfiles:
    print(a)
    c = mypath + a
    nome = c.split('/')[-1].split('.')[0].split('_')[-1]
    print('name ' + nome)
    hist_file = np.load(c,allow_pickle=True)
    for b in hist_file.files:
        print('key '+ b)
        bins = hist_file[b][1].tolist()
        counts = hist_file[b][0].tolist()
        if b == 'trans':
            trans_hist[nome] = { 'bin' : bins, 'counts': counts}
        elif b == 'long':
            long_hist[nome] = { 'bin' : bins, 'counts': counts}
        else: print('error in POL')
        assert len(bins) == len(counts) + 1
##part fos single pol hist
newmypath = '/Users/it058990/Desktop/MicheleG/PhD/VBS_COST_Group/GITBicocca/DNN_storage/evaluation/ful_lep_VBS_WWmuvev_phantom_1_6_nob/cos_theta_sel4/all_hist_toplot/sel_hist/'
onlyfiles = [f for f in listdir(newmypath) if isfile(join(newmypath, f))]
onlyfiles

for a in onlyfiles:
    print(a)
    c = newmypath + a
    nome = c.split('/')[-1].split('.')[0].split('_')[-2]
    print('name ' + nome)
    pol = c.split('/')[-1].split('.')[0].split('_')[-1]
    print('pol ' + pol)
    hist_file = np.load(c,allow_pickle=True)
    b = hist_file.files[0]
    print('key '+ b)
    bins = hist_file[b][1].tolist()
    counts = hist_file[b][0].tolist()
    if pol == 'trans':
        trans_hist[nome] = { 'bin' : bins, 'counts': counts}
    elif pol == 'long':
        long_hist[nome] = { 'bin' : bins, 'counts': counts}
    else: print('error in POL')
    assert len(bins) == len(counts) + 1
#save long and trans dict to dataframe and then to disk
df_tr = pd.DataFrame(trans_hist)
df_ln = pd.DataFrame(long_hist)

import pickle 
try: 
    trans_file = open('trans_file', 'wb') 
    pickle.dump(df_tr, trans_file) 
    trans_file.close() 
  
except: 
    print("Something went wrong")

try: 
    long_file = open('long_file', 'wb') 
    pickle.dump(df_ln, long_file) 
    long_file.close() 
  
except: 
    print("Something went wrong")
# #PLOTTING    
# for i in trans_hist.keys():
#     print(i)
#     bins = trans_hist[i]['bin']
#     counts = trans_hist[i]['counts']
#     print(len(bins))
#     print(len(counts))

#     plt.figure(1)
#     plt.legend()
#     plt.hist(bins[:-1], bins, weights=counts, density = True)
#     plt.show()
