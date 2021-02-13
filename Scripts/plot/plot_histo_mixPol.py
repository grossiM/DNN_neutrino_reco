#!/usr/bin/env python3
"""

  Michele Grossi <michele.grossi@it.ibm.com>
  Jakob Novak <janob.novak.cern.ch>
  Version 1.0,  September 2020

  USAGE: python3  plot_evaluated_mixPol.py -c JobOption/NNplot_config.cfg
  """

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
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
args = parser.parse_args()

config = configparser.ConfigParser()
config.optionxform = str
config.read(args.config)


#this part should be implemented if the for cicle to change the folder name according to all selection list
where_save =  config.get('output','output-folder')

if not os.path.exists(where_save):
    os.system('mkdir ' + where_save)


ful_comp = bool(int(config.get('plotting','ful-comp')))
print(ful_comp)
print(type(ful_comp))

###data reading & checking
####hdf5 reading
hdf_mix = pd.read_hdf(config.get('input','data-ev'))
#truth_label = config.get('input','truth-label')

if config.get('selection','zero-delta-only') == '1':
    hdf_mix = hdf_mix.query("mu_delta==0")

#################################################################selection and removal
to_rm = config.get('selection','discard').split(',')
print(to_rm)
###
for model_to_rm in to_rm:
    #bad_mod = fnmatch.filter(hdf_mix.columns, '*'+ model_to_rm + '*')
    bad_mod = fnmatch.filter(hdf_mix.columns, model_to_rm)
    print('discarded branches:')
    print(bad_mod)
    hdf_mix = hdf_mix.drop(bad_mod,axis=1)
#################################################################plotting
binning = config.get('plotting', 'binning')
binning = binning.replace('\\', '')
bins = binning.split(',')
b1,b2,b3 = float(bins[0]), float(bins[1]), float(bins[2])


good = []
for wildcard in config.get('selection','wildcard').split(','):
    print('BINARY')
    print('wildcard: ',wildcard)
    print('match: ',fnmatch.filter(hdf_mix.columns,wildcard))
    good = good + fnmatch.filter(hdf_mix.columns,wildcard)
    print('good: ',good)


hdf_mix_l = hdf_mix[hdf_mix['pol_label'] == 1]
print('ORIGINAL LONG_DIMENSION: ',hdf_mix_l.shape[0])

hdf_mix_t = hdf_mix[hdf_mix['pol_label'] == 0]
print('ORIGINAL TRANSVERSE_DIMENSION: ',hdf_mix_t.shape[0])


""""""
#the unpolarized sample does not have a truth so I need to plot the score (not rounded)
def plot_bin(name, where, random=False):

    entry = name.split('_')[0].replace('bat128','')
    print('ENTRY: ',entry)
    if (config.get('plotting', 'normalize') == '1'):
        normalize = True
    else:
        normalize = False

    a = name.strip('_e100')
    b = a+ '_rounded_score'

    if (config.get('plotting', 'invert') == '1'):
        score = hdf_mix[name]
    else:
        score = hdf_mix[name]
    print('DIMENSION: ',len(score))
    plt.figure(1)
    h_long = plt.hist(score, np.arange(b1, b2, b3), label=entry, density=normalize, histtype='step', linewidth=2)
    np.savez(where + '/h_' + name, long=h_long)

""""""
""""""
def plot_bin_l(name, where, random=False):

    entry = name.split('_')[0].replace('bat128','')
    print('ENTRY: ',entry)
    if (config.get('plotting', 'normalize') == '1'):
        normalize = True
    else:
        normalize = False

    a = name.strip('_e100')
    b = a+ '_rounded_score'

    if (config.get('plotting', 'invert') == '1'):
        score = hdf_mix[name][hdf_mix['pol_label'] == 0]
    else:
        score = hdf_mix[name][hdf_mix['pol_label'] == 1]
    print('DIMENSION: ',len(score))
    plt.figure(1)
    h_long = plt.hist(score, np.arange(b1, b2, b3), label=entry, density=normalize, histtype='step', linewidth=2)
    np.savez(where + '/h_' + name +'_LONG', long=h_long)

""""""

if (config.get('plotting', 'normalize') == '1'):
    normalize = True
else:
    normalize = False

#here create the figure
#######LONGITUDINAL
fig = plt.figure(1)

#############looping through selected model
print('looping through selected models:')
for c in good:
    print(c)
    if ful_comp:
        plot_bin(c,where_save)
    else:
        print('LONG PLOT')
        plot_bin_l(c,where_save)
print('plotting executed')

reco_type = 'classification'

#plt.figure(1)
art_l = []
lgd_l = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1),ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_l.append(lgd_l)
plt.xlabel(config.get('plotting','xlabel'))
plt.ylabel('Number of events')
ymin, ymax = plt.ylim()
plt.ylim((ymin,1.1*ymax))
#plt.annotate('W full computation',xy=(-0.8, ymax),fontsize=14,weight='bold')

if ful_comp:
    pol = 'full_comp'
else:
    pol = 'LONG'
fig.savefig(where_save + '/score_' + pol +'.pdf', additional_artists=art_l,bbox_inches="tight")

copyfile(args.config, where_save+ '/thisconfig.cfg')
print('figures saved into '+where_save)
