#!/usr/bin/env python3
"""

  Michele Grossi <michele.grossi@it.ibm.com>
  Jakob Novak <janob.novak.cern.ch>
  Version 1.0,  September 2020

  USAGE: python3  plot_evaluated_multilabelPol.py -c JobOption/NNplot_config.cfg
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
print('will save in: ',where_save)

if os.path.exists(where_save+'/theta_trans.pdf') or \
   os.path.exists(where_save+'/theta_long.pdf'):
    raise ValueError('plots in the '+where_save+' would be overwritten, exiting')
if not os.path.exists(where_save):
    os.system('mkdir ' + where_save)


cos2_long = {}
cos2_mix = {}
cos2_trans = {}
###data reading & checking
####hdf5 reading
hdf_mix = pd.read_hdf(config.get('input','data-ev'))
##### REORDER COLUMNS
hdf_mix = hdf_mix.reindex(sorted(hdf_mix.columns), axis=1)
#### CHECK NAN
###
mix_rm = hdf_mix.columns[hdf_mix.isna().any()].tolist()
print('Models containing Nan: \n {0}'.format(mix_rm))
hdf_mix = hdf_mix.drop(mix_rm,axis=1)
hdf_mix = hdf_mix.reindex(sorted(hdf_mix.columns), axis=1)

######################################selection and removal
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
    print('MULTIBINARY')
    #   DEVO SELEZIONARE SOLO LA BASE DEL NOME DEL MODELLO E POI LE SINGOLE FUNZIONI AGGIUNGONO IL PEZZO +'__cat0_rounded_score'
    #wildcard += '_cat0_rounded_score'               
    print('wildcard: ',wildcard)
    print('match: ',fnmatch.filter(hdf_mix.columns,'*'+wildcard+'*_cat0_rounded_score'))
    good = good + fnmatch.filter(hdf_mix.columns,'*'+wildcard+'*_cat0_rounded_score')
    print('good: ',good)

#ora devo definire la truth per longlong, transtrans, mixpol mutually exclusive
hdf_mix_l = hdf_mix[hdf_mix['longitudinal'] == 1]
print('ORIGINAL LONG_DIMENSION: ',hdf_mix_l.shape[0])

hdf_mix_m = hdf_mix[hdf_mix['mixed'] == 1]
print('ORIGINAL MIX_DIMENSION: ',hdf_mix_m.shape[0])

hdf_mix_t = hdf_mix[hdf_mix['transverse'] == 1]
print('ORIGINAL TRANSVERSE_DIMENSION: ',hdf_mix_t.shape[0])

""""""
def plot_bin_ll(name, where, random=False):

    entry = name.replace('bat128','')
    print('ENTRY: ',entry)
    
    if (config.get('plotting', 'normalize') == '1'):
        normalize = True
    else:
        normalize = False
    new_name = name +'_cat0_rounded_score'
    #long_tmp = hdf_mix[hdf_mix[new_name]==1]
    cos_l_mu = hdf_mix['mu_truth_cos_theta'][hdf_mix[new_name]==1]
    cos_l_el = hdf_mix['el_truth_cos_theta'][hdf_mix[new_name]==1]

    #cos2_long[new_name] = pd.concat([long_tmp['mu_truth_cos_theta'],long_tmp['mu_truth_cos_theta']])
    cos2_long[new_name] = pd.concat([cos_l_mu,cos_l_el])

    plt.figure(1)
    h_long = plt.hist(cos2_long[new_name].values, np.arange(b1, b2, b3), label=entry, density=normalize, histtype='step', linewidth=2)
    np.savez(where + '/h_' + name+ '_LONG', long=h_long)

""""""
def plot_bin_mix(name, where, random=False):

    entry = name.replace('bat128','')
    print('ENTRY: ',entry)

    if (config.get('plotting', 'normalize') == '1'):
        normalize = True
    else:
        normalize = False
    new_name = name +'_cat1_rounded_score'
    cos_mix_mu = hdf_mix['mu_truth_cos_theta'][hdf_mix[new_name]==1]
    cos_mix_el = hdf_mix['el_truth_cos_theta'][hdf_mix[new_name]==1]

    cos2_mix[new_name] = pd.concat([cos_mix_mu,cos_mix_el])
    plt.figure(2)
    h_mix = plt.hist(cos2_mix[new_name].values, np.arange(b1, b2, b3), label=entry, density=normalize, histtype='step', linewidth=2)
    np.savez(where + '/h_' + name+ '_MIX', long=h_mix)
""""""
def plot_bin_tt(name, where, random=False):

    entry = name.replace('bat128','')
    print('ENTRY: ',entry)

    if (config.get('plotting', 'normalize') == '1'):
        normalize = True
    else:
        normalize = False
    new_name = name +'_cat2_rounded_score'
    cos_t_mu = hdf_mix['mu_truth_cos_theta'][hdf_mix[new_name]==1]
    cos_t_el = hdf_mix['el_truth_cos_theta'][hdf_mix[new_name]==1]

    cos2_trans[new_name] = pd.concat([cos_t_mu,cos_t_el])
    plt.figure(3)
    h_trans = plt.hist(cos2_trans[new_name].values, np.arange(b1, b2, b3), label=entry, density=normalize, histtype='step', linewidth=2)

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
    cos2_long['truth_cos_thetas'] = pd.concat([hdf_mix_l['mu_truth_cos_theta'],hdf_mix_l['el_truth_cos_theta']])
    h_long_true = plt.hist(cos2_long['truth_cos_thetas'],np.arange(b1, b2, b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='Truth Longitudinal')
    
#######Mixed
fig_mix = plt.figure(2)
if (config.get('plotting', 'truth') == '1'):
    cos2_trans['truth_cos_thetas'] = pd.concat([hdf_mix_m['mu_truth_cos_theta'],hdf_mix_m['el_truth_cos_theta']])
    h_mix_true = plt.hist(cos2_trans['truth_cos_thetas'],np.arange(b1, b2, b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='Truth Transverse')
    
# #########transverse
fig_trans = plt.figure(3)
if (config.get('plotting', 'truth') == '1'):
    cos2_trans['truth_cos_thetas'] = pd.concat([hdf_mix_t['mu_truth_cos_theta'],hdf_mix_t['el_truth_cos_theta']])
    h_trans_true = plt.hist(cos2_trans['truth_cos_thetas'],np.arange(b1, b2, b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='Truth Transverse')
    
# ########################   saving all truth things
if (config.get('plotting', 'truth') == '1'):
    np.savez(where_save + '/h_truth', trans=h_trans_true, long=h_long_true)
# ####################################

#############looping through selected model
print('looping through selected models:')
for c in good:
    print(c)
    d = c.split('_')[0]
    plot_bin_ll(d,where_save)
    plot_bin_mix(d,where_save)
    plot_bin_tt(d,where_save)
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
plt.annotate(r'W$_{\mathbf{LL}}$ polarization',xy=(-0.8, ymax),fontsize=14,weight='bold')


plt.figure(2)
art_m = []
lgd_m = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1),ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_m.append(lgd_m)
#plt.title('Mixed polarization, '+reco_type)
plt.xlabel(config.get('plotting','xlabel'))
#plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')
ymin, ymax = plt.ylim()
plt.ylim((ymin,1.1*ymax))
plt.annotate(r'W$_{\mathbf{LT}}$ polarization',xy=(-0.8, ymax),fontsize=14,weight='bold')


plt.figure(3)
art_t = []
lgd_t = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_t.append(lgd_t)
#plt.title('Transverse polarization, '+reco_type)
plt.xlabel(config.get('plotting','xlabel'))
#plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')
ymin, ymax = plt.ylim()
plt.ylim((ymin,1.1*ymax))
plt.annotate(r'W$_{\mathbf{TT}}$ polarization',xy=(-0.8, ymax),fontsize=14,weight='bold')


fig_long.savefig(where_save + '/theta_long.pdf', additional_artists=art_l,bbox_inches="tight")
fig_mix.savefig(where_save + '/theta_mix.pdf', additional_artists=art_m,bbox_inches="tight")
fig_trans.savefig(where_save + '/theta_trans.pdf', additional_artists=art_t,bbox_inches="tight")

copyfile(args.config, where_save+ '/thisconfig.cfg')
print('figures saved into '+where_save)
