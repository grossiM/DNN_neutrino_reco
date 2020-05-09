#!/usr/bin/env python3
#################################
# M. Grossi - J.Novak #2019
################################
#macro to plot npz files with specific name convention
#usage: python3 plot_hist.py -c plt_hist.cfg
########################

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

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
args = parser.parse_args()

config = configparser.ConfigParser()
config.optionxform = str
config.read(args.config)

where_save =  config.get('output','output-folder')
#for time being it is hardcoded...this can be changed in future implementation
hist_key = 'arr_0'
#read all npz file, create histogram, take the name from the filename and use legend

def plot_single(hist_0,hist_key,n_sel):

    try:
        hist_0[hist_key]
        print(f'{hist_key} is ok.')
    except KeyError:
        print(f"{hist_key} is not a file in the archive.")
    
    if (config.get('plotting', 'normalize') == '1'):
        normalize = True
    else:
        normalize = False

    entry = 'selection '+ str(n_sel)
    plt.figure(1)
    plt.legend()
    plt.hist(hist_0[hist_key], label=entry, density=normalize, histtype='step', linewidth=2)

for c in config.get('input','data-list').split(':'):
    
    hist_file = np.load(c)
    sample_orig = c.split('/')[-1]
    selection = sample_orig.split('.')[0].split('_')[-1] 
    n_sel = [int(s) for s in selection if s.isdigit()] 
    if len(n_sel)==0:
        n_sel = 'random'
    else:
        n_sel = n_sel[0]
    plot_single(hist_file,hist_key,n_sel)
print('plotting executed and saved in: '+ where_save)


plt.figure(1)
art_l = []
lgd_l = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1),ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_l.append(lgd_l)
plt.xlabel('test')
plt.ylabel('Number of events')
plt.savefig(where_save + '/test.pdf', additional_artists=art_l,bbox_inches="tight")

