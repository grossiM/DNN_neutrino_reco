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

#manca la parte per fare plot score nel caso di full computation
ful_lep = False

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
mix_rm = hdf_mix.columns[hdf_mix.isna().any()].tolist()
print('Models containing Nan: \n {0}'.format(mix_rm))
hdf_mix = hdf_mix.drop(mix_rm,axis=1)
hdf_mix = hdf_mix.reindex(sorted(hdf_mix.columns), axis=1)

######################################selection and removal
to_rm = config.get('selection','discard').split(',')
print(to_rm)
### no removal of bad model for ful comp
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



wildcard = config.get('selection','wildcard')
print('MULTIBINARY')
#wildcard += '_cat0_rounded_score'               
print('wildcard: ',wildcard)
good = fnmatch.filter(hdf_mix.columns,'*'+wildcard+'*_cat0_rounded_score')
print('good: ',good)

#now define 3 separate truth dataset to evaluate score longlong, transtrans, mixpol mutually exclusive
hdf_mix_l = hdf_mix[hdf_mix['longitudinal'] == 1]
print('ORIGINAL LONG_DIMENSION: ',hdf_mix_l.shape[0])

hdf_mix_m = hdf_mix[hdf_mix['mixed'] == 1]
print('ORIGINAL MIX_DIMENSION: ',hdf_mix_m.shape[0])

hdf_mix_t = hdf_mix[hdf_mix['transverse'] == 1]
print('ORIGINAL TRANSVERSE_DIMENSION: ',hdf_mix_t.shape[0])

""""""
#plot the 3 score for the given model
def plot_bin_ll(name, where, random=False):

    entry = name.replace('bat128','')
    print('ENTRY: ',entry)
    
    if (config.get('plotting', 'normalize') == '1'):
        normalize = True
    else:
        normalize = False
    round_name = name +'_cat0_rounded_score'
    score_name0 = name +'_cat0_e100'
    score_name1 = name +'_cat1_e100'
    score_name2 = name +'_cat2_e100'

    scorel0 = hdf_mix_l[score_name0]
    scorel1 = hdf_mix_l[score_name1]
    scorel2 = hdf_mix_l[score_name2]
    print('min value LL: ',min(hdf_mix_l[score_name0]))
    print('min value LT: ',min(hdf_mix_l[score_name1]))
    print('min value TT: ',min(hdf_mix_l[score_name1]))
    
    plt.figure(1)
    h_long = plt.hist(scorel0,  np.arange(b1, b2, b3), label='LL', density=normalize, histtype='step', linewidth=2)
    h_long = plt.hist(scorel1, np.arange(b1, b2, b3), label='LT', density=normalize, histtype='step', linewidth=2)
    h_long = plt.hist(scorel2, np.arange(b1, b2, b3), label='TT', density=normalize, histtype='step', linewidth=2)
    np.savez(where + '/h_' + name+ '_LONG', long=h_long)

""""""
def plot_bin_mix(name, where, random=False):

    entry = name.replace('bat128','')
    print('ENTRY: ',entry)

    if (config.get('plotting', 'normalize') == '1'):
        normalize = True
    else:
        normalize = False

    round_name = name +'_cat1_rounded_score'
    score_name0 = name +'_cat0_e100'
    score_name1 = name +'_cat1_e100'
    score_name2 = name +'_cat2_e100'

    scorem0 = hdf_mix_m[score_name0]
    scorem1 = hdf_mix_m[score_name1]
    scorem2 = hdf_mix_m[score_name2]

    plt.figure(2)
    h_mix = plt.hist(scorem0, np.arange(b1, b2, b3), label='LL', density=normalize, histtype='step', linewidth=2)
    h_mix = plt.hist(scorem1, np.arange(b1, b2, b3), label='LT', density=normalize, histtype='step', linewidth=2)
    h_mix = plt.hist(scorem2, np.arange(b1, b2, b3), label='TT', density=normalize, histtype='step', linewidth=2)

    np.savez(where + '/h_' + name+ '_MIX', long=h_mix)
""""""
def plot_bin_tt(name, where, random=False):

    entry = name.replace('bat128','')
    print('ENTRY: ',entry)

    if (config.get('plotting', 'normalize') == '1'):
        normalize = True
    else:
        normalize = False
    round_name = name +'_cat2_rounded_score'
    score_name0 = name +'_cat0_e100'
    score_name1 = name +'_cat1_e100'
    score_name2 = name +'_cat2_e100'

    scoret0 = hdf_mix_t[score_name0]
    scoret1 = hdf_mix_t[score_name1]
    scoret2 = hdf_mix_t[score_name2]

    plt.figure(3)
    h_trans = plt.hist(scoret0, np.arange(b1, b2, b3), label='LL', density=normalize, histtype='step', linewidth=2)
    h_trans = plt.hist(scoret1, np.arange(b1, b2, b3), label='LT', density=normalize, histtype='step', linewidth=2)
    h_trans = plt.hist(scoret2, np.arange(b1, b2, b3),label='TT', density=normalize, histtype='step', linewidth=2)

    np.savez(where + '/h_' + name + '_TRANS', trans=h_trans)

""""""
def plot_bin_fulcomp(name, where, random=False):

    entry = name.replace('bat128','')
    print('ENTRY: ',entry)

    if (config.get('plotting', 'normalize') == '1'):
        normalize = True
    else:
        normalize = False
    score_name0 = name +'_cat0_e100'
    score_name1 = name +'_cat1_e100'
    score_name2 = name +'_cat2_e100'

    score0 = hdf_mix[score_name0]
    score1 = hdf_mix[score_name1]
    score2 = hdf_mix[score_name2]

    plt.figure(4)
    h_unpol = plt.hist(score0, np.arange(b1, b2, b3), label='LL', density=normalize, histtype='step', linewidth=2)
    h_unpol = plt.hist(score1, np.arange(b1, b2, b3), label='LT', density=normalize, histtype='step', linewidth=2)
    h_unpol = plt.hist(score2, np.arange(b1, b2, b3),label='TT', density=normalize, histtype='step', linewidth=2)

    np.savez(where + '/h_' + name + '_fulcomp', unpol=h_unpol)
""""""

print('PLOT selected models:')

d = good[0].split('_')[0]
if ful_lep:
    plot_bin_fulcomp(d,where_save)

    fig_fulcomp = plt.figure(4)

    art_f = []
    lgd_f = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1),ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
    art_f.append(lgd_f)
    plt.xlabel('scores')
    plt.ylabel('Number of events')
    ymin, ymax = plt.ylim()
    plt.ylim((ymin,1.1*ymax))
    plt.annotate('Full Computation',xy=(0.4, ymax),fontsize=14,weight='bold')
    fig_fulcomp.savefig(where_save + '/theta_fulcomp.pdf', additional_artists=art_f, bbox_inches="tight")

else:
        #############looping through selected model
    plot_bin_ll(d,where_save)
    plot_bin_mix(d,where_save)
    plot_bin_tt(d,where_save)
    print('plotting executed')
    ################################
    fig_long = plt.figure(1)

    art_l = []
    lgd_l = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1),ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
    art_l.append(lgd_l)
    plt.xlabel('LL score')
    plt.ylabel('Number of events')
    ymin, ymax = plt.ylim()
    plt.ylim((ymin,1.1*ymax))
    plt.annotate(r'W$_{\mathbf{LL}}$ polarization',xy=(0.4, ymax),fontsize=14,weight='bold')


    fig_mix = plt.figure(2)
    art_m = []
    lgd_m = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1),ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
    art_m.append(lgd_m)
    #plt.title('Mixed polarization, '+reco_type)
    plt.xlabel('TL score')
    plt.ylabel('Number of events')
    ymin, ymax = plt.ylim()
    plt.ylim((ymin,1.1*ymax))
    plt.annotate(r'W$_{\mathbf{LT}}$ polarization',xy=(0.4, ymax),fontsize=14,weight='bold')


    fig_trans = plt.figure(3)
    art_t = []
    lgd_t = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
    art_t.append(lgd_t)
    #plt.title('Transverse polarization, '+reco_type)
    plt.xlabel('TT score')
    plt.ylabel('Number of events')
    ymin, ymax = plt.ylim()
    plt.ylim((ymin,1.1*ymax))
    plt.annotate(r'W$_{\mathbf{TT}}$ polarization',xy=(0.4, ymax),fontsize=14,weight='bold')

    fig_long.savefig(where_save + '/theta_long.pdf', additional_artists=art_l, bbox_inches="tight")
    fig_mix.savefig(where_save + '/theta_mix.pdf', additional_artists=art_m,bbox_inches="tight")
    fig_trans.savefig(where_save + '/theta_trans.pdf', additional_artists=art_t,bbox_inches="tight")

copyfile(args.config, where_save+ '/thisconfig.cfg')
print('figures saved into '+where_save)
