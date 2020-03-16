#!/usr/bin/env python3
"""

  Michele Grossi <michele.grossi@it.ibm.com>
  Jakob Novak <janob.novak.cern.ch>
  Version 1.0, February 2020

  USAGE: python3  plot_evaluated.py -c NNplot_config.cfg
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

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
args = parser.parse_args()

config = configparser.ConfigParser()
config.optionxform = str
config.read(args.config)

#this part should be implemented if the for cicle to change the folder name according to all selection list
where_save =  config.get('output','output-folder')

if os.path.exists(where_save+'/theta_full.pdf') or \
   os.path.exists(where_save+'/theta_unpol.pdf') or \
   os.path.exists(where_save+'/theta_trans.pdf') or \
   os.path.exists(where_save+'/theta_long.pdf'):
    raise ValueError('plots in the '+where_save+'would be overwritten, exiting')
if not os.path.exists(where_save):
    os.system('mkdir ' + where_save)



###data reading & checking
####hdf5 reading
hdf_long = pd.read_hdf(config.get('input','data-long'))
hdf_trans = pd.read_hdf(config.get('input','data-trans'))
hdf_unpol = pd.read_hdf(config.get('input','data-unpol'))
hdf_full_comp = pd.read_hdf(config.get('input','data-fulcomp'))

######
avlb_data = np.zeros((4, 1), dtype=bool)
try:
    hdf_long, hdf_trans, hdf_unpol, hdf_full_comp
    avlb_data[0] = True
    avlb_data[1] = True
    avlb_data[2] = True
    avlb_data[3] = True
    pol_list = ['long','trans','unpol','fullcomp']
except NameError:
    print('not all polarized calibrated sample provided')


#selection and removal
to_rm = config.get('selection','discard').split(',')

###
for model_to_rm in to_rm:
    bad_mod = fnmatch.filter(hdf_long.columns, '*'+ model_to_rm + '*')
    print('bad_mod')
    print(bad_mod)
    hdf_long = hdf_long.drop(bad_mod,axis=1)
    hdf_trans.drop(bad_mod,axis=1)
    hdf_unpol.drop(bad_mod,axis=1)
    hdf_full_comp.drop(bad_mod,axis=1)
###

#plotting
binning = config.get('plotting', 'binning')
binning = binning.replace('\\', '')
bins = binning.split(',')
b1,b2,b3 = float(bins[0]), float(bins[1]), float(bins[2])


good = []
for wildcard in config.get('selection','wildcard').split(','):
    if config.get('selection','type') == 'binary':
        wildcard += '_rounded_score'
    #elif config.get('selection','type') == 'regression':
        #wildcard += '_pred'
    good = good + fnmatch.filter(hdf_long.columns,wildcard)

if config.get('selection','type') == 'binary':

    s_l = hdf_long[['sol0_cos_theta','sol1_cos_theta']].values
    s_t = hdf_trans[['sol0_cos_theta','sol1_cos_theta']].values
    s_u = hdf_unpol[['sol0_cos_theta','sol1_cos_theta']].values
    s_f = hdf_full_comp[['sol0_cos_theta','sol1_cos_theta']].values
    

""""""
def plot_bin(name, avlb_pol, where):

    for pol_type in avlb_pol:

        pattern = config.get('legend','entry').split(':')
        entry = re.sub(pattern[0],pattern[1], name.rstrip())

        if pol_type == 'long':
            score_l = hdf_long[name]
            cos_l = [s_l[i, sign] for i, sign in enumerate(score_l)]
            plt.figure(1)
            h_long = plt.hist(cos_l, np.arange(b1, b2, b3), label=entry, density=True, histtype='step', linewidth=2)

        elif pol_type == 'trans':
            score_t = hdf_trans[name]
            cos_t = [s_t[i, sign] for i, sign in enumerate(score_t)]
            plt.figure(2)
            h_trans = plt.hist(cos_t, np.arange(b1, b2, b3), label=entry, density=True, histtype='step', linewidth=2)

        elif pol_type == 'unpol':
            score_u = hdf_unpol[name]
            cos_u = [s_u[i, sign] for i, sign in enumerate(score_u)]
            plt.figure(3)
            h_unpol = plt.hist(cos_u, np.arange(b1, b2, b3), label=entry, density=True, histtype='step', linewidth=2)

        elif pol_type == 'fullcomp':
            score_f = hdf_full_comp[name]
            cos_f = [s_f[i, sign] for i, sign in enumerate(score_f)]
            plt.figure(4)
            h_full = plt.hist(cos_f, np.arange(b1, b2, b3), label=entry, density=True, histtype='step', linewidth=2)

        else:
            print('wrong polarization')
    np.savez(where + '/h_' + name, unpol=h_unpol, trans=h_trans, long=h_long, fulcomp = h_full)

""""""

""""""
def plot_reg(name,avlb_pol, where):

    for pol_type in avlb_pol:

        if pol_type == 'long':
            plt.figure(1)
            plt.legend()
            h_long = plt.hist(hdf_long[name].values, np.arange(b1, b2, b3),alpha=0.3, label=name.split('_')[0], density=True, histtype='step', linewidth=2)

        elif pol_type == 'trans':
            plt.figure(2)
            plt.legend()
            h_trans = plt.hist(hdf_trans[name].values, np.arange(b1, b2, b3),alpha=0.3, label=name.split('_')[0], density=True, histtype='step', linewidth=2)

        elif pol_type == 'unpol':
            plt.figure(3)
            plt.legend()
            h_unpol = plt.hist(hdf_unpol[name].values, np.arange(b1, b2, b3),alpha=0.3, label=name.split('_')[0], density=True, histtype='step', linewidth=2)

        elif pol_type == 'fullcomp':
            plt.figure(4)
            h_full = plt.hist(hdf_full_comp[name].values, np.arange(b1, b2, b3),alpha=0.3, label=name.split('_')[0], density=True, histtype='step', linewidth=2)

        else:
            print('wrong polarization')
    np.savez(where + '/h_' + name, unpol=h_unpol, trans=h_trans, long=h_long, fulcomp = h_full)

""""""
#here create the figure
#######LONGITUDINAL
fig_long = plt.figure(1)
h_long_true = plt.hist(hdf_long['truth_cos_theta'],np.arange(b1, b2, b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=True, linewidth=2, label='truth')

# #########transverse
fig_trans = plt.figure(2)
h_trans_true = plt.hist(hdf_trans['truth_cos_theta'], np.arange(b1,b2,b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=True, linewidth=2, label='truth')

# #######unpolarized
fig_unpol = plt.figure(3)
h_unpol_true = plt.hist(hdf_unpol['truth_cos_theta'], np.arange(b1,b2,b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=True, linewidth=2, label='truth')

# ######full computation
fig_full = plt.figure(4)
h_fullcomp_true = plt.hist(hdf_full_comp['truth_cos_theta'], np.arange(b1,b2,b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=True, linewidth=2, label='truth')

# ########################   saving all truth things
np.savez(where_save + '/h_truth',unpol=h_unpol_true, trans=h_trans_true, long=h_long_true, fulcomp = h_fullcomp_true)
# ####################################

#############looping through selected model
print('looping through selected models:')

for c in good:
    #here implement check if binary or regression! o sopra
    print(c)
    if config.get('selection','type') == 'binary':
        plot_bin(c,pol_list,where_save)
    elif config.get('selection','type') == 'regression':
        plot_reg(c,pol_list,where_save)
    else:
        raise ValueError('Error: wrong evaluation type selected')
print('plotting executed')

plt.figure(1)
plt.legend(loc='upper left', ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
plt.title('Longitudinal polarization')
plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')
plt.ylim((0, 1.2*plt.ylim()[1]))

plt.figure(2)
plt.legend(loc='upper left', ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
plt.title('Transverse polarization')
plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')
plt.ylim((0, 1.2*plt.ylim()[1]))

plt.figure(3)
plt.legend(loc='upper left', ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
plt.title('Unpolarized OSP')
plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')
plt.ylim((0, 1.2*plt.ylim()[1]))

plt.figure(4)
plt.legend(loc='upper left', ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
plt.title('Full computation')
plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')
plt.ylim((0, 1.2*plt.ylim()[1]))

fig_long.savefig(where_save + '/theta_long.pdf')
fig_trans.savefig(where_save + '/theta_trans.pdf')
fig_unpol.savefig(where_save + '/theta_unpol.pdf')
fig_full.savefig(where_save + '/theta_full.pdf')

print('figures saved into '+where_save)
