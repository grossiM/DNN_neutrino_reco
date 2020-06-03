#!/usr/bin/env python3
"""

  Michele Grossi <michele.grossi@it.ibm.com>
  Jakob Novak <janob.novak.cern.ch>
  Version 1.0, June 2020

  USAGE: python3  plot_ful_lep.py -c JobOption/***_config.cfg
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
    raise ValueError('plots in the '+where_save+' would be overwritten, exiting')
if not os.path.exists(where_save):
    os.system('mkdir ' + where_save)



###data reading & checking
####hdf5 reading, for time being only trans and long component are present
hdf_long = pd.read_hdf(config.get('input','data-long'))
hdf_trans = pd.read_hdf(config.get('input','data-trans'))
#hdf_unpol = pd.read_hdf(config.get('input','data-unpol'))
#hdf_full_comp = pd.read_hdf(config.get('input','data-fulcomp'))

######
avlb_data = np.zeros((2, 1), dtype=bool) #put 4 instead of2 in case all 4 samples are present
try:
    hdf_long, hdf_trans#, hdf_unpol, hdf_full_comp
    avlb_data[0] = True
    avlb_data[1] = True
    #avlb_data[2] = True
    #avlb_data[3] = True
    pol_list = ['long','trans']
    #pol_list = ['long','trans','unpol','fullcomp']
except NameError:
    print('not all polarized calibrated sample provided')


#################################################################selection and removal
to_rm = config.get('selection','discard').split(',')
print(to_rm)
###
for model_to_rm in to_rm:
    #bad_mod = fnmatch.filter(hdf_long.columns, '*'+ model_to_rm + '*')
    bad_mod = fnmatch.filter(hdf_long.columns, model_to_rm)
    print('discarded branches:')
    print(bad_mod)
    hdf_long = hdf_long.drop(bad_mod,axis=1)
    hdf_trans.drop(bad_mod,axis=1)
    #hdf_unpol.drop(bad_mod,axis=1)
    #hdf_full_comp.drop(bad_mod,axis=1)
###

#################################################################plotting
binning = config.get('plotting', 'binning')
binning = binning.replace('\\', '')
bins = binning.split(',')
b1,b2,b3 = float(bins[0]), float(bins[1]), float(bins[2])


good = []
for wildcard in config.get('selection','wildcard').split(','):
    if config.get('selection','type') == 'regression':
        wildcard += '_pred'
    good = good + fnmatch.filter(hdf_long.columns,wildcard)
#case where regression predict 6 neutrino components
if config.get('selection','type') == 'reg_neutrinos':
#mapping from training config: mapping={'cat0': 'v_mu_px', 'cat1': 'v_mu_py','cat2': 'v_mu_pz','cat3': 'v_el_px', 'cat4': 'v_el_py', 'cat5': 'v_el_pz'}
    lep_cat_list = ['cat0','cat1','cat3', 'cat4']
    ##########long
    #muon
    mu_px_list_long = (fnmatch.filter(hdf_long.columns, '*'+lep_cat_list[0]+'*'))
    mu_py_list_long= (fnmatch.filter(hdf_long.columns, '*'+lep_cat_list[1]+'*'))
    #electron
    el_px_list_long = (fnmatch.filter(hdf_long.columns, '*'+lep_cat_list[2]+'*'))
    el_py_list_long = (fnmatch.filter(hdf_long.columns, '*'+lep_cat_list[3]+'*'))
    ##########trans
    #muon
    mu_px_list_trans = (fnmatch.filter(hdf_trans.columns, '*'+lep_cat_list[0]+'*'))
    mu_py_list_trans= (fnmatch.filter(hdf_trans.columns, '*'+lep_cat_list[1]+'*'))
    #electron
    el_px_list_trans = (fnmatch.filter(hdf_trans.columns, '*'+lep_cat_list[2]+'*'))
    el_py_list_trans = (fnmatch.filter(hdf_trans.columns, '*'+lep_cat_list[3]+'*'))

    #file manipulation
    #long
    for i,j,k,l in zip(mu_px_list_long,mu_py_list_long,el_px_list_long,el_px_list_long):
        hdf_long[i[:-10]+'_v_mu_pt'] = np.sqrt(hdf_long[i]**2 + hdf_long[j]**2)
        hdf_long[i[:-10]+'_v_el_pt'] = np.sqrt(hdf_long[k]**2 + hdf_long[l]**2)
    #trans
    for p,q,r,s in zip(mu_px_list_trans,mu_py_list_trans,el_px_list_trans,el_px_list_trans):
        hdf_trans[p[:-10]+'_v_mu_pt'] = np.sqrt(hdf_trans[p]**2 + hdf_trans[q]**2)
        hdf_trans[p[:-10]+'_v_el_pt'] = np.sqrt(hdf_trans[r]**2 + hdf_trans[s]**2)

""""""
def plot_multireg(name,avlb_pol, where):

    for pol_type in avlb_pol:

        pattern = config.get('legend','entry').split(':')
        entry = re.sub(pattern[0],pattern[1], name.rstrip())
        
        entry = re.sub('_e100', '', entry)

        if (config.get('plotting', 'normalize') == '1'):
            normalize = True
        else:
            normalize = False

        if pol_type == 'long':
            plt.figure(1)
            plt.legend()
            h_long = plt.hist(hdf_long[name].values, np.arange(b1, b2, b3), label=entry, density=normalize, histtype='step', linewidth=2)

        elif pol_type == 'trans':
            plt.figure(2)
            plt.legend()
            h_trans = plt.hist(hdf_trans[name].values, np.arange(b1, b2, b3), label=entry, density=normalize, histtype='step', linewidth=2)

        # elif pol_type == 'unpol':
        #     plt.figure(3)
        #     plt.legend()
        #     h_unpol = plt.hist(hdf_unpol[name].values, np.arange(b1, b2, b3), label=entry, density=normalize, histtype='step', linewidth=2)

        # elif pol_type == 'fullcomp':
        #     plt.figure(4)
        #     h_full = plt.hist(hdf_full_comp[name].values, np.arange(b1, b2, b3), label=entry, density=normalize, histtype='step', linewidth=2)

        else:
            print('wrong polarization')
    np.savez(where + '/h_' + name, trans=h_trans, long=h_long)
    #np.savez(where + '/h_' + name, unpol=h_unpol, trans=h_trans, long=h_long, fulcomp = h_full)

""""""

if (config.get('plotting', 'normalize') == '1'):
    normalize = True
else:
    normalize = False

#here create the figure
#######LONGITUDINAL
fig_long = plt.figure(1)
if (config.get('plotting', 'truth') == '1'):
    if config.get('selection','type') == 'reg_neutrinos':
        #implement truth pt v_mu_pt
        h_long_true = plt.hist([hdf_long['v_mu_pt'],hdf_long['v_el_pt']],np.arange(b1, b2, b3), stacked=True, histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='mu Truth')#check label
       
    elif config.get('selection','type') == 'regcostheta':
        h_long_true = plt.hist(hdf_long['mu_truth_cos_theta'],np.arange(b1, b2, b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='mu Truth')
        h_long_true = plt.hist(hdf_long['el_truth_cos_theta'],np.arange(b1, b2, b3), histtype='stepfilled', facecolor='w', hatch='+', edgecolor='C0', density=normalize, linewidth=2, label='el Truth')

# #########transverse
fig_trans = plt.figure(2)
if (config.get('plotting', 'truth') == '1'):
    if config.get('selection','type') == 'reg_neutrinos':
        #implement truth pt v_mu_pt
        h_trans_true = plt.hist([hdf_trans['v_mu_pt'],hdf_trans['v_el_pt']],np.arange(b1, b2, b3), stacked=True, histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='mu Truth')#check label
       
    elif config.get('selection','type') == 'regcostheta':
        h_trans_true = plt.hist(hdf_trans['mu_truth_cos_theta'],np.arange(b1, b2, b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='mu Truth')
        h_trans_true = plt.hist(hdf_trans['el_truth_cos_theta'],np.arange(b1, b2, b3), histtype='stepfilled', facecolor='w', hatch='+', edgecolor='C0', density=normalize, linewidth=2, label='el Truth')

# # #######unpolarized
# fig_unpol = plt.figure(3)
# if (config.get('plotting', 'truth') == '1'):
#     h_unpol_true = plt.hist(hdf_unpol['truth_cos_theta'], np.arange(b1,b2,b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='Truth')

# # ######full computation
# fig_full = plt.figure(4)
# if (config.get('plotting', 'truth') == '1'):
#     h_fullcomp_true = plt.hist(hdf_full_comp['truth_cos_theta'], np.arange(b1,b2,b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='Truth')

# ########################   saving all truth things
if (config.get('plotting', 'truth') == '1'):
    np.savez(where_save + '/h_truth',trans=h_trans_true, long=h_long_true)
    #np.savez(where_save + '/h_truth',unpol=h_unpol_true, trans=h_trans_true, long=h_long_true, fulcomp = h_fullcomp_true)
# ####################################

#############looping through selected model
print('looping through selected models:')
  
for c in good:
    #here implement check if binary or regression!
    print(c)
    if config.get('selection','type') == 'regcostheta':
        plot_multireg(c,pol_list,where_save)
    else:
        raise ValueError('Error: wrong evaluation type selected')
print('plotting executed')


plt.figure(1)
art_l = []
lgd_l = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1),ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_l.append(lgd_l)
#plt.title('Longitudinal polarization, '+reco_type)
plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')
#plt.ylim((0, 1.2*plt.ylim()[1]))
# plt.ylim((0, 1.2))

plt.figure(2)
art_t = []
lgd_t = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_t.append(lgd_t)
#plt.title('Transverse polarization, '+reco_type)
plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')
# plt.ylim((0, 1.2))
#plt.ylim((0, 1.2*plt.ylim()[1]))

# plt.figure(3)
# art_u = []
# lgd_u = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
# art_u.append(lgd_u)
# #plt.title('Unpolarized OSP, '+reco_type)
# plt.xlabel('cos'+r'$\theta$')
# plt.ylabel('Number of events')
# #plt.ylim((0, 1.2*plt.ylim()[1]))
# # plt.ylim((0, 1.2))

# plt.figure(4)
# art_f = []
# lgd_f = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
# art_f.append(lgd_f)
# #plt.title('Full computation, '+reco_type)
# plt.xlabel('cos'+r'$\theta$')
# plt.ylabel('Number of events')
# #plt.ylim((0, 1.2*plt.ylim()[1]))
# # plt.ylim((0, 1.2))

fig_long.savefig(where_save + '/theta_long.pdf', additional_artists=art_l,bbox_inches="tight")
fig_trans.savefig(where_save + '/theta_trans.pdf', additional_artists=art_t,bbox_inches="tight")
#fig_unpol.savefig(where_save + '/theta_unpol.pdf', additional_artists=art_u,bbox_inches="tight")
#fig_full.savefig(where_save + '/theta_full.pdf', additional_artists=art_f,bbox_inches="tight")

print('figures saved into '+where_save)
