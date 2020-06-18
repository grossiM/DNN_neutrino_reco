#!/usr/bin/env python3
"""

  Michele Grossi <michele.grossi@it.ibm.com>
  Jakob Novak <janob.novak.cern.ch>
  Version 1.0, March 2020

  USAGE: python3  NNplot_compare.py -c JobObtion/NNplot_compareREG.cfg
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

from sklearn.metrics import roc_auc_score, roc_curve

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
    raise ValueError('comparison plots in the '+where_save+' would be overwritten, exiting')
if not os.path.exists(where_save):
    os.system('mkdir ' + where_save)



###data reading & checking
####hdf5 reading
#######10k
hdf_long_10k = pd.read_hdf(config.get('input','data-long_10k'))
hdf_trans_10k = pd.read_hdf(config.get('input','data-trans_10k'))
#####100k
hdf_long_100k = pd.read_hdf(config.get('input','data-long_100k'))
hdf_trans_100k = pd.read_hdf(config.get('input','data-trans_100k'))
######1M
hdf_long_1M = pd.read_hdf(config.get('input','data-long_1M'))
hdf_trans_1M = pd.read_hdf(config.get('input','data-trans_1M'))
######10M
hdf_long_10M = pd.read_hdf(config.get('input','data-long_10M'))
hdf_trans_10M = pd.read_hdf(config.get('input','data-trans_10M'))
######
pol_list = ['trans','long']

#plotting
binning = config.get('plotting', 'binning')
binning = binning.replace('\\', '')
bins = binning.split(',')
b1,b2,b3 = float(bins[0]), float(bins[1]), float(bins[2])

""""""
def plot_reg(name,avlb_pol, where):

    for pol_type in avlb_pol:

        if pol_type == 'long':
            plt.figure(1)
            plt.legend()
            h_long = plt.hist(np.concatenate((hdf_long_10k[name+'bat128dro0.0_cat0_e100'].values,hdf_long_10k[name+'bat128dro0.0_cat1_e100'].values)), np.arange(b1, b2, b3), label='direct, no MAOS', histtype='step', linewidth=2)
            h_long = plt.hist(np.concatenate((hdf_long_100k[name+'bat64_cat0_cos'].values,hdf_long_100k[name+'bat64_cat3_cos'].values)), np.arange(b1, b2, b3), label='indirect, no MAOS', histtype='step', linewidth=2)
            h_long = plt.hist(np.concatenate((hdf_long_1M[name+'bat64_cat0_e100'].values,hdf_long_1M[name+'bat64_cat1_e100'].values)), np.arange(b1, b2, b3), label='direct, with MAOS', histtype='step', linewidth=2)
            h_long = plt.hist(np.concatenate((hdf_long_10M[name+'bat256cat0_cos'].values,hdf_long_10M[name+'bat256cat3_cos'].values)), np.arange(b1, b2, b3), label='indirect, with MAOS', histtype='step', linewidth=2)

        elif pol_type == 'trans':
            fig_test = plt.figure(2)
            h_trans = plt.hist(np.concatenate((hdf_trans_10k[name+'bat128dro0.0_cat0_e100'].values,hdf_trans_10k[name+'bat128dro0.0_cat1_e100'].values)), np.arange(b1, b2, b3), label='direct, no MAOS', histtype='step', linewidth=2)
            h_trans = plt.hist(np.concatenate((hdf_trans_100k[name+'bat64_cat0_cos'].values,hdf_trans_100k[name+'bat64_cat3_cos'].values)), np.arange(b1, b2, b3), label='indirect, no MAOS', histtype='step', linewidth=2)
            h_trans = plt.hist(np.concatenate((hdf_trans_1M[name+'bat64_cat0_e100'].values,hdf_trans_1M[name+'bat64_cat1_e100'].values)), np.arange(b1, b2, b3), label='direct, with MAOS', histtype='step', linewidth=2)
            h_trans = plt.hist(np.concatenate((hdf_trans_10M[name+'bat256cat0_cos'].values,hdf_trans_10M[name+'bat256cat3_cos'].values)), np.arange(b1, b2, b3), label='indirect, with MAOS', histtype='step', linewidth=2)
            plt.legend()

        else:
            print('wrong polarization')

""""""

if (config.get('plotting', 'normalize') == '1'):
    normalize = True
else:
    normalize = False

#here create the figure
#######LONGITUDINAL
fig_long = plt.figure(1)
h_long_true = plt.hist(np.concatenate([hdf_long_1M['mu_truth_cos_theta'].values,hdf_long_1M['el_truth_cos_theta'].values]),np.arange(b1, b2, b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='truth')

# #########transverse
fig_trans = plt.figure(2)
h_trans_true = plt.hist(np.concatenate([hdf_trans_1M['mu_truth_cos_theta'],hdf_trans_1M['el_truth_cos_theta']]), np.arange(b1,b2,b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='truth')

# ########################   saving all truth things
np.savez(where_save + '/h_truth', trans=h_trans_true, long=h_long_true)
# ####################################

model = config.get('selection','model')
print('plotting model '+model)

plot_reg(model,pol_list,where_save)
print('plotting executed')

##bbox_to_anchor to be defined in config, if needed particular format

plt.figure(1)
art_l = []
lgd_l = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_l.append(lgd_l)
#plt.title('Longitudinal polarization, '+reco_type)
plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')
ymin, ymax = plt.ylim()
plt.ylim((ymin,1.1*ymax))
plt.annotate(r'W$_{\mathbf{L}}$ polarization',xy=(-0.8, ymax),fontsize=14,weight='bold')

plt.figure(2)
art_t = []
lgd_t = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_t.append(lgd_t)
#plt.title('Transverse polarization, '+reco_type)
plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')
ymin, ymax = plt.ylim()
plt.ylim((ymin,1.1*ymax))
plt.annotate(r'W$_{\mathbf{T}}$ polarization',xy=(-0.8, ymax),fontsize=14,weight='bold')


fig_long.savefig(where_save + '/theta_long.pdf', additional_artists=art_l,bbox_inches="tight")
fig_trans.savefig(where_save + '/theta_trans.pdf', additional_artists=art_t,bbox_inches="tight")

print('figures saved into '+where_save)
