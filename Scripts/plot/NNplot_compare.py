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
    raise ValueError('comparison plots in the '+where_save+' would be overwritten, exiting')
if not os.path.exists(where_save):
    os.system('mkdir ' + where_save)



###data reading & checking
####hdf5 reading
#######500
hdf_long_500 = pd.read_hdf(config.get('input','data-long_500'))
hdf_trans_500 = pd.read_hdf(config.get('input','data-trans_500'))
hdf_unpol_500 = pd.read_hdf(config.get('input','data-unpol_500'))
hdf_full_comp_500 = pd.read_hdf(config.get('input','data-fulcomp_500'))
#####10k
hdf_long_10k = pd.read_hdf(config.get('input','data-long_10k'))
hdf_trans_10k = pd.read_hdf(config.get('input','data-trans_10k'))
hdf_unpol_10k = pd.read_hdf(config.get('input','data-unpol_10k'))
hdf_full_comp_10k = pd.read_hdf(config.get('input','data-fulcomp_10k'))
######1M
hdf_long_1M = pd.read_hdf(config.get('input','data-long_1M'))
hdf_trans_1M = pd.read_hdf(config.get('input','data-trans_1M'))
hdf_unpol_1M = pd.read_hdf(config.get('input','data-unpol_1M'))
hdf_full_comp_1M = pd.read_hdf(config.get('input','data-fulcomp_1M'))
######10M
hdf_long_10M = pd.read_hdf(config.get('input','data-long_10M'))
hdf_trans_10M = pd.read_hdf(config.get('input','data-trans_10M'))
hdf_unpol_10M = pd.read_hdf(config.get('input','data-unpol_10M'))
hdf_full_comp_10M = pd.read_hdf(config.get('input','data-fulcomp_10M'))
######
pol_list = ['long','trans','unpol','fullcomp']

#selection and removal
to_rm = config.get('selection','discard').split(',')
print('DISCARD:')
print('*'*50)
print(to_rm)
print('*'*50)
###
for model_to_rm in to_rm:
    bad_mod_500 = fnmatch.filter(hdf_long_500.columns, '*'+ model_to_rm + '*')
    bad_mod_10k = fnmatch.filter(hdf_long_10k.columns, '*'+ model_to_rm + '*')
    bad_mod_1M = fnmatch.filter(hdf_long_1M.columns, '*'+ model_to_rm + '*')
    bad_mod_10M = fnmatch.filter(hdf_long_10M.columns, '*'+ model_to_rm + '*')

    print('discarded branches 500:')
    print(bad_mod_500)
    print('discarded branches 10k:')
    print(bad_mod_10k)
    print('discarded branches 1M:')
    print(bad_mod_1M)
    print('discarded branches 10M:')
    print(bad_mod_10M)
    #500
    hdf_long_500 = hdf_long_500.drop(bad_mod_500,axis=1)
    hdf_trans_500.drop(bad_mod_500,axis=1)
    hdf_unpol_500.drop(bad_mod_500,axis=1)
    hdf_full_comp_500.drop(bad_mod_500,axis=1)
    #10k
    hdf_long_10k = hdf_long_10k.drop(bad_mod_10k,axis=1)
    hdf_trans_10k.drop(bad_mod_10k,axis=1)
    hdf_unpol_10k.drop(bad_mod_10k,axis=1)
    hdf_full_comp_10k.drop(bad_mod_10k,axis=1)
    #1M
    hdf_long_1M = hdf_long_1M.drop(bad_mod_1M,axis=1)
    hdf_trans_1M.drop(bad_mod_1M,axis=1)
    hdf_unpol_1M.drop(bad_mod_1M,axis=1)
    hdf_full_comp_1M.drop(bad_mod_1M,axis=1)
    #10M
    hdf_long_10M = hdf_long_10M.drop(bad_mod_10M,axis=1)
    hdf_trans_10M.drop(bad_mod_10M,axis=1)
    hdf_unpol_10M.drop(bad_mod_10M,axis=1)
    hdf_full_comp_10M.drop(bad_mod_10M,axis=1)
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
    good = good + fnmatch.filter(hdf_long_500.columns,wildcard)
    
if config.get('selection','type') == 'binary':

    #500
    s_l_500 = hdf_long_500[['sol0_cos_theta','sol1_cos_theta']].values
    s_t_500 = hdf_trans_500[['sol0_cos_theta','sol1_cos_theta']].values
    s_u_500 = hdf_unpol_500[['sol0_cos_theta','sol1_cos_theta']].values
    s_f_500 = hdf_full_comp_500[['sol0_cos_theta','sol1_cos_theta']].values
    #10k
    s_l_10k = hdf_long_10k[['sol0_cos_theta','sol1_cos_theta']].values
    s_t_10k = hdf_trans_10k[['sol0_cos_theta','sol1_cos_theta']].values
    s_u_10k = hdf_unpol_10k[['sol0_cos_theta','sol1_cos_theta']].values
    s_f_10k = hdf_full_comp_10k[['sol0_cos_theta','sol1_cos_theta']].values

    #1M
    s_l_1M = hdf_long_1M[['sol0_cos_theta','sol1_cos_theta']].values
    s_t_1M = hdf_trans_1M[['sol0_cos_theta','sol1_cos_theta']].values
    s_u_1M = hdf_unpol_1M[['sol0_cos_theta','sol1_cos_theta']].values
    s_f_1M = hdf_full_comp_1M[['sol0_cos_theta','sol1_cos_theta']].values

    #10M
    s_l_10M = hdf_long_10M[['sol0_cos_theta','sol1_cos_theta']].values
    s_t_10M = hdf_trans_10M[['sol0_cos_theta','sol1_cos_theta']].values
    s_u_10M = hdf_unpol_10M[['sol0_cos_theta','sol1_cos_theta']].values
    s_f_10M = hdf_full_comp_10M[['sol0_cos_theta','sol1_cos_theta']].values

""""""
def plot_bin(name, avlb_pol, where):

    for pol_type in avlb_pol:

        pattern = config.get('legend','entry').split(':')
        entry = re.sub(pattern[0],pattern[1], name.rstrip())

        if pol_type == 'long':
            #500
            score_l_500 = hdf_long_500[name]
            cos_l_500 = [s_l_500[i, sign] for i, sign in enumerate(score_l_500)]
            #10k
            score_l_10k = hdf_long_10k[name]
            cos_l_10k = [s_l_10k[i, sign] for i, sign in enumerate(score_l_10k)]
            #1M
            score_l_1M = hdf_long_1M[name]
            cos_l_1M = [s_l_1M[i, sign] for i, sign in enumerate(score_l_1M)]
            #10M
            score_l_10M = hdf_long_10M[name]
            cos_l_10M = [s_l_10M[i, sign] for i, sign in enumerate(score_l_10M)]
            
            plt.figure(1)
            h_long = plt.hist(cos_l_500, np.arange(b1, b2, b3), label=entry+'_500', density=True, histtype='step', linewidth=2)
            h_long = plt.hist(cos_l_10k, np.arange(b1, b2, b3), label=entry+'_10k', density=True, histtype='step', linewidth=2)
            h_long = plt.hist(cos_l_1M, np.arange(b1, b2, b3), label=entry+'_1M', density=True, histtype='step', linewidth=2)
            h_long = plt.hist(cos_l_10M, np.arange(b1, b2, b3), label=entry+'_10M', density=True, histtype='step', linewidth=2)


        elif pol_type == 'trans':
            #500
            score_t_500 = hdf_trans_500[name]
            cos_t_500 = [s_t_500[i, sign] for i, sign in enumerate(score_t_500)]
            #10k
            score_t_10k = hdf_trans_10k[name]
            cos_t_10k = [s_t_10k[i, sign] for i, sign in enumerate(score_t_10k)]
            #1M
            score_t_1M = hdf_trans_1M[name]
            cos_t_1M = [s_t_1M[i, sign] for i, sign in enumerate(score_t_1M)]
            #10M
            score_t_10M = hdf_trans_10M[name]
            cos_t_10M = [s_t_10M[i, sign] for i, sign in enumerate(score_t_10M)]

            plt.figure(2)
            h_trans = plt.hist(cos_t_500, np.arange(b1, b2, b3), label=entry+'_500', density=True, histtype='step', linewidth=2)
            h_trans = plt.hist(cos_t_10k, np.arange(b1, b2, b3), label=entry+'_10k', density=True, histtype='step', linewidth=2)
            h_trans = plt.hist(cos_t_1M, np.arange(b1, b2, b3), label=entry+'_1M', density=True, histtype='step', linewidth=2)
            h_trans = plt.hist(cos_t_10M, np.arange(b1, b2, b3), label=entry+'_10M', density=True, histtype='step', linewidth=2)



        elif pol_type == 'unpol':
            #500
            score_u_500 = hdf_unpol_500[name]
            cos_u_500 = [s_u_500[i, sign] for i, sign in enumerate(score_u_500)]
            #10k
            score_u_10k = hdf_unpol_10k[name]
            cos_u_10k = [s_u_10k[i, sign] for i, sign in enumerate(score_u_10k)]
            #1M
            score_u_1M = hdf_unpol_1M[name]
            cos_u_1M = [s_u_1M[i, sign] for i, sign in enumerate(score_u_1M)]
            #10M
            score_u_10M = hdf_unpol_10M[name]
            cos_u_10M = [s_u_10M[i, sign] for i, sign in enumerate(score_u_10M)]

            plt.figure(3)
            h_unpol = plt.hist(cos_u_500, np.arange(b1, b2, b3), label=entry+'_500', density=True, histtype='step', linewidth=2)
            h_unpol = plt.hist(cos_u_10k, np.arange(b1, b2, b3), label=entry+'_10k', density=True, histtype='step', linewidth=2)
            h_unpol = plt.hist(cos_u_1M, np.arange(b1, b2, b3), label=entry+'_1M', density=True, histtype='step', linewidth=2)
            h_unpol = plt.hist(cos_u_10M, np.arange(b1, b2, b3), label=entry+'_10M', density=True, histtype='step', linewidth=2)

        elif pol_type == 'fullcomp':
            #500
            score_f_500 = hdf_full_comp_500[name]
            cos_f_500 = [s_f_500[i, sign] for i, sign in enumerate(score_f_500)]
            #10k
            score_f_10k = hdf_full_comp_10k[name]
            cos_f_10k = [s_f_10k[i, sign] for i, sign in enumerate(score_f_10k)]
            #1M
            score_f_1M = hdf_full_comp_1M[name]
            cos_f_1M = [s_f_1M[i, sign] for i, sign in enumerate(score_f_1M)]
            #10M
            score_f_10M = hdf_full_comp_10M[name]
            cos_f_10M = [s_f_10M[i, sign] for i, sign in enumerate(score_f_10M)]

            plt.figure(4)
            h_full = plt.hist(cos_f_500, np.arange(b1, b2, b3), label=entry+'_500', density=True, histtype='step', linewidth=2)
            h_full = plt.hist(cos_f_10k, np.arange(b1, b2, b3), label=entry+'_10k', density=True, histtype='step', linewidth=2)
            h_full = plt.hist(cos_f_1M, np.arange(b1, b2, b3), label=entry+'_1M', density=True, histtype='step', linewidth=2)
            h_full = plt.hist(cos_f_10M, np.arange(b1, b2, b3), label=entry+'_10M', density=True, histtype='step', linewidth=2)

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
            h_long = plt.hist(hdf_long_500[name].values, np.arange(b1, b2, b3), label=name.split('_')[0].replace('bat32', '')+'_500', density=True, histtype='step', linewidth=2)
            h_long = plt.hist(hdf_long_10k[name].values, np.arange(b1, b2, b3), label=name.split('_')[0].replace('bat32', '')+'_10k', density=True, histtype='step', linewidth=2)
            h_long = plt.hist(hdf_long_1M[name].values, np.arange(b1, b2, b3), label=name.split('_')[0].replace('bat32', '')+'_1M', density=True, histtype='step', linewidth=2)
            h_long = plt.hist(hdf_long_10M[name.replace('bat32', '')].values, np.arange(b1, b2, b3), label=name.split('_')[0].replace('_e100', '').replace('bat32', '')+'_10M', density=True, histtype='step', linewidth=2)

        elif pol_type == 'trans':
            plt.figure(2)
            plt.legend()
            h_trans = plt.hist(hdf_trans_500[name].values, np.arange(b1, b2, b3), label=name.split('_')[0].replace('bat32', '')+'_500', density=True, histtype='step', linewidth=2)
            h_trans = plt.hist(hdf_trans_10k[name].values, np.arange(b1, b2, b3), label=name.split('_')[0].replace('bat32', '')+'_10k', density=True, histtype='step', linewidth=2)
            h_trans = plt.hist(hdf_trans_1M[name].values, np.arange(b1, b2, b3), label=name.split('_')[0].replace('bat32', '')+'_1M', density=True, histtype='step', linewidth=2)
            h_trans = plt.hist(hdf_trans_10M[name.replace('bat32', '')].values, np.arange(b1, b2, b3), label=name.split('_')[0].replace('_e100', '').replace('bat32', '')+'_10M', density=True, histtype='step', linewidth=2)

        elif pol_type == 'unpol':
            plt.figure(3)
            plt.legend()
            h_unpol = plt.hist(hdf_unpol_500[name].values, np.arange(b1, b2, b3), label=name.split('_')[0].replace('bat32', '')+'_500', density=True, histtype='step', linewidth=2)
            h_unpol = plt.hist(hdf_unpol_10k[name].values, np.arange(b1, b2, b3), label=name.split('_')[0].replace('bat32', '')+'_10k', density=True, histtype='step', linewidth=2)
            h_unpol = plt.hist(hdf_unpol_1M[name].values, np.arange(b1, b2, b3), label=name.split('_')[0].replace('bat32', '')+'_1M', density=True, histtype='step', linewidth=2)
            h_unpol = plt.hist(hdf_unpol_10M[name.replace('bat32', '')].values, np.arange(b1, b2, b3), label=name.split('_')[0].replace('_e100', '').replace('bat32', '')+'10M', density=True, histtype='step', linewidth=1)

        elif pol_type == 'fullcomp':
            plt.figure(4)
            h_full = plt.hist(hdf_full_comp_500[name].values, np.arange(b1, b2, b3), label=name.split('_')[0].replace('bat32', '')+'_500', density=True, histtype='step', linewidth=2)
            h_full = plt.hist(hdf_full_comp_10k[name].values, np.arange(b1, b2, b3), label=name.split('_')[0].replace('bat32', '')+'_10k', density=True, histtype='step', linewidth=2)
            h_full = plt.hist(hdf_full_comp_1M[name].values, np.arange(b1, b2, b3), label=name.split('_')[0].replace('bat32', '')+'_1M', density=True, histtype='step', linewidth=2)
            h_full = plt.hist(hdf_full_comp_10M[name.replace('bat32', '')].values, np.arange(b1, b2, b3), label=name.split('_')[0].replace('_e100', '').replace('bat32', '')+'_10M', density=True, histtype='step', linewidth=2)


        else:
            print('wrong polarization')
    np.savez(where + '/h_' + name, unpol=h_unpol, trans=h_trans, long=h_long, fulcomp = h_full)

""""""
#here create the figure
#######LONGITUDINAL
fig_long = plt.figure(1)
h_long_true = plt.hist(hdf_long_1M['truth_cos_theta'],np.arange(b1, b2, b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=True, linewidth=2, label='truth')

# #########transverse
fig_trans = plt.figure(2)
h_trans_true = plt.hist(hdf_trans_1M['truth_cos_theta'], np.arange(b1,b2,b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=True, linewidth=2, label='truth')

# #######unpolarized
fig_unpol = plt.figure(3)
h_unpol_true = plt.hist(hdf_unpol_1M['truth_cos_theta'], np.arange(b1,b2,b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=True, linewidth=2, label='truth')

# ######full computation
fig_full = plt.figure(4)
h_fullcomp_true = plt.hist(hdf_full_comp_1M['truth_cos_theta'], np.arange(b1,b2,b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=True, linewidth=2, label='truth')

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

if config.get('selection','type') == 'regression':
    reco_type = 'regression'
else :
    reco_type = 'classification'

##bbox_to_anchor to be defined in config, if needed particular format

plt.figure(1)
art_l = []
lgd_l = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_l.append(lgd_l)
plt.title('Longitudinal polarization, '+reco_type)
plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')
plt.ylim((0, 1.2*plt.ylim()[1]))

plt.figure(2)
art_t = []
lgd_t = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_t.append(lgd_t)
plt.title('Transverse polarization, '+reco_type)
plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')
plt.ylim((0, 1.2*plt.ylim()[1]))

plt.figure(3)
art_u = []
lgd_u = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_u.append(lgd_u)
plt.title('Unpolarized OSP, '+reco_type)
plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')
plt.ylim((0, 1.2*plt.ylim()[1]))

plt.figure(4)
art_f = []
lgd_f = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_f.append(lgd_f)
plt.title('Full computation, '+reco_type)
plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')
plt.ylim((0, 1.2*plt.ylim()[1]))

fig_long.savefig(where_save + '/theta_long.pdf', additional_artists=art_l,bbox_inches="tight")
fig_trans.savefig(where_save + '/theta_trans.pdf', additional_artists=art_t,bbox_inches="tight")
fig_unpol.savefig(where_save + '/theta_unpol.pdf', additional_artists=art_u,bbox_inches="tight")
fig_full.savefig(where_save + '/theta_full.pdf', additional_artists=art_f,bbox_inches="tight")

print('figures saved into '+where_save)
