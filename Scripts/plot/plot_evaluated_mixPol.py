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

if os.path.exists(where_save+'/theta_trans.pdf') or \
   os.path.exists(where_save+'/theta_long.pdf'):
    raise ValueError('plots in the '+where_save+' would be overwritten, exiting')
if not os.path.exists(where_save):
    os.system('mkdir ' + where_save)



###data reading & checking
####hdf5 reading
hdf_mix = pd.read_hdf(config.get('input','data-ev'))
truth_label = config.get('input','truth-label')

if config.get('selection','zero-delta-only') == '1':
    hdf_mix = hdf_mix.query("mu_delta==0")

#################################################################selection and removal
to_rm = config.get('selection','discard').split(',')
print(to_rm)
###
for model_to_rm in to_rm:
    bad_mod = fnmatch.filter(hdf_mix.columns, '*'+ model_to_rm + '*')
    #bad_mod = fnmatch.filter(hdf_mix.columns, model_to_rm)
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
    wildcard += '_rounded_score'               # to train on autoai output or SELECTION CRITERIA
    print('wildcard: ',wildcard)
    print('match: ',fnmatch.filter(hdf_mix.columns,wildcard))
    good = good + fnmatch.filter(hdf_mix.columns,wildcard)
    print('good: ',good)

#ora devo definire la truth per long e trans
hdf_mix_l = hdf_mix[hdf_mix['pol_label'] == 1]
print('ORIGINAL LONG_DIMENSION: ',hdf_mix_l.shape[0])

hdf_mix_t = hdf_mix[hdf_mix['pol_label'] == 0]
print('ORIGINAL TRANSVERSE_DIMENSION: ',hdf_mix_t.shape[0])

""""""
def plot_bin_l(name, where, random=False):

    entry = name.split('_')[0].replace('bat128','')
    print('ENTRY: ',entry)
    if (config.get('plotting', 'normalize') == '1'):
        normalize = True
    else:
        normalize = False

    if (config.get('plotting', 'invert') == '1'):
        cos_l = hdf_mix[truth_label][hdf_mix[name]==0]
    else:
        cos_l = hdf_mix[truth_label][hdf_mix[name]==1]
    print('LONG_DIMENSION: ',len(cos_l))
    plt.figure(1)
    h_long = plt.hist(cos_l, np.arange(b1, b2, b3), label=entry, density=normalize, histtype='step', linewidth=2)
    np.savez(where + '/h_' + name+ '_LONG', long=h_long)

""""""
def plot_bin_t(name, where, random=False):

    
    entry = name.split('_')[0].replace('bat128','')
    if (config.get('plotting', 'normalize') == '1'):
        normalize = True
    else:
        normalize = False

    if (config.get('plotting', 'invert') == '1'):
        cos_t = hdf_mix[truth_label][hdf_mix[name]==1]
    else:
        cos_t = hdf_mix[truth_label][hdf_mix[name]==0]
    print('TRANS_DIMENSION: ',len(cos_t))
    plt.figure(2)
    h_trans = plt.hist(cos_t, np.arange(b1, b2, b3), label=entry, density=normalize, histtype='step', linewidth=2)

    np.savez(where + '/h_' + name + '_TRANS', trans=h_trans)

""""""
""""""

if (config.get('plotting', 'normalize') == '1'):
    normalize = True
else:
    normalize = False

#here create the figure
#######LONGITUDINAL
fig_long = plt.figure(1)
if (config.get('plotting', 'truth') == '1'):
    h_long_true = plt.hist(hdf_mix_l['truth_cos_theta'],np.arange(b1, b2, b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='Truth')

# #########transverse
fig_trans = plt.figure(2)
if (config.get('plotting', 'truth') == '1'):
    h_trans_true = plt.hist(hdf_mix_t['truth_cos_theta'], np.arange(b1,b2,b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='Truth')

# ########################   saving all truth things
if (config.get('plotting', 'truth') == '1'):
    np.savez(where_save + '/h_truth', trans=h_trans_true, long=h_long_true)
# ####################################

#############looping through selected model
print('looping through selected models:')
for c in good:
    print(c)
    plot_bin_l(c,where_save)
    plot_bin_t(c,where_save)
print('plotting executed')

reco_type = 'classification'

plt.figure(1)
art_l = []
lgd_l = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1),ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_l.append(lgd_l)
#plt.title('Longitudinal polarization, '+reco_type)
plt.xlabel(config.get('plotting','xlabel'))
#plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')
ymin, ymax = plt.ylim()
plt.ylim((ymin,1.1*ymax))
plt.annotate(r'W$_{\mathbf{L}}$ polarization',xy=(-0.8, ymax),fontsize=14,weight='bold')
#plt.ylim((0, 1.2*plt.ylim()[1]))
# plt.ylim((0, 1.2))

plt.figure(2)
art_t = []
lgd_t = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_t.append(lgd_t)
#plt.title('Transverse polarization, '+reco_type)
plt.xlabel(config.get('plotting','xlabel'))
#plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')
ymin, ymax = plt.ylim()
plt.ylim((ymin,1.1*ymax))
plt.annotate(r'W$_{\mathbf{T}}$ polarization',xy=(-0.8, ymax),fontsize=14,weight='bold')
# plt.ylim((0, 1.2))
#plt.ylim((0, 1.2*plt.ylim()[1]))

fig_long.savefig(where_save + '/theta_long.pdf', additional_artists=art_l,bbox_inches="tight")
fig_trans.savefig(where_save + '/theta_trans.pdf', additional_artists=art_t,bbox_inches="tight")

copyfile(args.config, where_save+ '/thisconfig.cfg')
print('figures saved into '+where_save)
