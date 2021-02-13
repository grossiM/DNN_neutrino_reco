#!/usr/bin/env python3
"""
  Michele Grossi <michele.grossi@it.ibm.com>
  Jakob Novak <janob.novak.cern.ch>
  Version 1.0, November 2020
  USAGE: python3  plot_ful_all_pol.py -c JobOption/***_config.cfg
  """
#plot direct cos theta distribution from ful leptonic mix polarisation that has all components

import os
import sys
import configparser
from shutil import copyfile
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
####hdf5 reading
hdf_long = pd.read_hdf(config.get('input','data-long'))
hdf_trans = pd.read_hdf(config.get('input','data-trans'))
hdf_mix = pd.read_hdf(config.get('input','data-mix'))
hdf_full_comp = pd.read_hdf(config.get('input','data-fulcomp'))

######
avlb_data = np.zeros((4, 1), dtype=bool)
try:
    hdf_long, hdf_trans, hdf_mix, hdf_full_comp
    avlb_data[0] = True
    avlb_data[1] = True
    avlb_data[2] = True
    avlb_data[3] = True
    pol_list = ['long','trans','mix','fullcomp']
except NameError:
    print('not all polarized calibrated sample provided')

hdf_long = hdf_long.reindex(sorted(hdf_long.columns), axis=1)
hdf_trans = hdf_trans.reindex(sorted(hdf_trans.columns), axis=1)
hdf_mix = hdf_mix.reindex(sorted(hdf_mix.columns), axis=1)
hdf_full_comp = hdf_full_comp.reindex(sorted(hdf_full_comp.columns), axis=1)
#print(hdf_long.columns)
#print(hdf_mix.columns)

#################################################################selection and removal
#test here
to_rm = config.get('selection','discard').split(',')
print(' \n model discarded: {0} '.format(str(to_rm)))

###
long_rm = hdf_long.columns[hdf_long.isna().any()].tolist()
print('Models containing Nan: \n {0}'.format(long_rm))
trans_rm = hdf_trans.columns[hdf_trans.isna().any()].tolist()
mix_rm = hdf_mix.columns[hdf_mix.isna().any()].tolist()
fulcomp_rm = hdf_full_comp.columns[hdf_full_comp.isna().any()].tolist()

hdf_long = hdf_long.drop(long_rm,axis=1)
hdf_long = hdf_long.reindex(sorted(hdf_long.columns), axis=1)

hdf_trans = hdf_trans.drop(trans_rm,axis=1)
hdf_trans = hdf_trans.reindex(sorted(hdf_trans.columns), axis=1)

hdf_mix = hdf_mix.drop(mix_rm,axis=1)
hdf_mix = hdf_mix.reindex(sorted(hdf_mix.columns), axis=1)

hdf_full_comp = hdf_full_comp.drop(fulcomp_rm,axis=1)
hdf_full_comp = hdf_full_comp.reindex(sorted(hdf_full_comp.columns), axis=1)

###
for model_to_rm in to_rm:
    #bad_mod = fnmatch.filter(hdf_long.columns, '*'+ model_to_rm + '*')
    bad_mod = fnmatch.filter(hdf_long.columns, model_to_rm)
    print('discarded branches:')
    print(bad_mod)
    hdf_long = hdf_long.drop(bad_mod,axis=1)
    hdf_trans.drop(bad_mod,axis=1)
    hdf_mix.drop(bad_mod,axis=1)
    hdf_full_comp.drop(bad_mod,axis=1)
###
#################################################################plotting
binning = config.get('plotting', 'binning')
binning = binning.replace('\\', '')
bins = binning.split(',')
b1,b2,b3 = float(bins[0]), float(bins[1]), float(bins[2])


good_mu = []
good_el = []
# good_mu_t = []
# good_el_t = []
# good_mu_m = []
# good_el_m = []
# good_mu_f = []
# good_el_f = []

cos2_long = {}
cos2_trans = {}
cos2_mix = {}
cos2_fulcomp = {}

for wildcard in config.get('selection','wildcard').split(','):

    print(' Plotting regression on costheta')
    #map cat0= el_truth_cos_theta, cat1= mu_truth_cos_theta --> neu60hid2bat64_cat0_e100  neu60hid2bat64_cat1_e100
    lep_cat_list = ['cat0','cat1']
    #### all
    #muon
    mu_cos_list = (fnmatch.filter(hdf_long.columns, '*'+lep_cat_list[1]+'*'))
    #electron
    el_cos_list = (fnmatch.filter(hdf_long.columns, '*'+lep_cat_list[0]+'*'))
    # ##########long
    # #muon
    # mu_cos_list_long = (fnmatch.filter(hdf_long.columns, '*'+lep_cat_list[1]+'*'))
    # #electron
    # el_cos_list_long = (fnmatch.filter(hdf_long.columns, '*'+lep_cat_list[0]+'*'))
    # ##########trans
    #     #muon
    # mu_cos_list_trans = (fnmatch.filter(hdf_trans.columns, '*'+lep_cat_list[1]+'*'))
    # #electron
    # el_cos_list_trans = (fnmatch.filter(hdf_trans.columns, '*'+lep_cat_list[0]+'*'))
    # ##########mix
    #     #muon
    # mu_cos_list_mix = (fnmatch.filter(hdf_mix.columns, '*'+lep_cat_list[1]+'*'))
    # #electron
    # el_cos_list_mix = (fnmatch.filter(hdf_mix.columns, '*'+lep_cat_list[0]+'*'))
    # ##########full computation
    #     #muon
    # mu_cos_list_fc = (fnmatch.filter(hdf_full_comp.columns, '*'+lep_cat_list[1]+'*'))
    # #electron
    # el_cos_list_fc = (fnmatch.filter(hdf_full_comp.columns, '*'+lep_cat_list[0]+'*'))


    good_mu = good_mu + fnmatch.filter(mu_cos_list,wildcard + '*')
    good_el = good_el + fnmatch.filter(el_cos_list,wildcard + '*')

    # good_mu_t = good_mu_t + fnmatch.filter(mu_cos_list_trans,wildcard + '*')
    # good_el_t = good_el_t + fnmatch.filter(el_cos_list_trans,wildcard + '*')

    # good_mu_m = good_mu_m + fnmatch.filter(mu_cos_list_mix,wildcard + '*')
    # good_el_m = good_el_m + fnmatch.filter(el_cos_list_mix,wildcard + '*')

    # good_mu_f = good_mu_f + fnmatch.filter(mu_cos_list_fc,wildcard + '*')
    # good_el_f = good_el_f + fnmatch.filter(el_cos_list_fc,wildcard + '*')
    

""""""
#def plot_multicos(name_el_l,name_mu_l,name_el_t,name_mu_t, name_el_m, name_mu_m,name_el_f,name_mu_f,avlb_pol, where):
def plot_multicos(name_el,name_mu,avlb_pol, where):

    for pol_type in avlb_pol:

        pattern = config.get('legend','entry').split(':')
        entry = re.sub(pattern[0],pattern[1], name_el.rstrip())
        
        entry = re.sub('_cat0_pred', '', entry)
        entry = re.sub('_cat0_e100','',entry)
        entry = entry.split('bat')[0]

        print(entry)
        if (config.get('plotting', 'normalize') == '1'):
            normalize = True
        else:
            normalize = False

        if pol_type == 'long':
            plt.figure(1)
            plt.legend()
            #cos2_long[name_el_l] = pd.DataFrame()
            cos2_long[name_el] = pd.concat([hdf_long[name_el],hdf_long[name_mu]])
            
            h_long = plt.hist(cos2_long[name_el].values, np.arange(b1, b2, b3), label=entry, density=normalize, histtype='step', linewidth=2)

        elif pol_type == 'trans':
            plt.figure(2)
            plt.legend()
            #cos2_trans[name_el_t] = pd.DataFrame()
            cos2_trans[name_el] = pd.concat([hdf_trans[name_el],hdf_trans[name_mu]])

            h_trans = plt.hist(cos2_trans[name_el].values, np.arange(b1, b2, b3), label=entry, density=normalize, histtype='step', linewidth=2)

        elif pol_type == 'mix':
            plt.figure(3)
            plt.legend()
            cos2_mix[name_el] = pd.concat([hdf_mix[name_el],hdf_mix[name_mu]])

            h_mix = plt.hist(cos2_mix[name_el].values, np.arange(b1, b2, b3), label=entry, density=normalize, histtype='step', linewidth=2)
        
        elif pol_type == 'fullcomp':
            plt.figure(4)
            plt.legend()
            cos2_fulcomp[name_el] = pd.concat([hdf_full_comp[name_el],hdf_full_comp[name_mu]])

            h_fc = plt.hist(cos2_fulcomp[name_el].values, np.arange(b1, b2, b3), label=entry, density=normalize, histtype='step', linewidth=2)


        else:
            print('wrong polarization')
    np.savez(where + '/h_' + entry, trans=h_trans, long=h_long, mix=h_mix, fulcomp=h_fc)

""""""
if (config.get('plotting', 'normalize') == '1'):
    normalize = True
else:
    normalize = False
##here create the figure
#######LONGITUDINAL
fig_long = plt.figure(1)
if (config.get('plotting', 'truth') == '1'):
    cos2_long['truth_cos_thetas'] = pd.concat([hdf_long['mu_truth_cos_theta'],hdf_long['el_truth_cos_theta']])
    h_long_true = plt.hist(cos2_long['truth_cos_thetas'],np.arange(b1, b2, b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='Truth Longitudinal')
# #########TRANSVERSE
fig_trans = plt.figure(2)
if (config.get('plotting', 'truth') == '1'):
    cos2_trans['truth_cos_thetas'] = pd.concat([hdf_trans['mu_truth_cos_theta'],hdf_trans['el_truth_cos_theta']])
    h_trans_true = plt.hist(cos2_trans['truth_cos_thetas'],np.arange(b1, b2, b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='Truth Transverse')
# #######mix
fig_mix = plt.figure(3)
if (config.get('plotting', 'truth') == '1'):
    cos2_mix['truth_cos_thetas'] = pd.concat([hdf_mix['mu_truth_cos_theta'],hdf_mix['el_truth_cos_theta']])
    h_mix_true = plt.hist(cos2_mix['truth_cos_thetas'], np.arange(b1,b2,b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='Truth')

# ######full computation
fig_full = plt.figure(4)
if (config.get('plotting', 'truth') == '1'):
    cos2_fulcomp['truth_cos_thetas'] = pd.concat([hdf_full_comp['mu_truth_cos_theta'],hdf_full_comp['el_truth_cos_theta']])
    h_fullcomp_true = plt.hist(cos2_fulcomp['truth_cos_thetas'], np.arange(b1,b2,b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='Truth')
  
    
# ########################   saving all truth things
if (config.get('plotting', 'truth') == '1'):
    np.savez(where_save + '/h_truth',trans=h_trans_true, long=h_long_true, mix=h_mix_true,fulcomp= h_fullcomp_true)
# ####################################

#############looping through selected model
print('looping through selected models:')
xlabel = 'cos'+r'$\theta$'
file_name = '/theta'
#for i,j,p,q,l,m,n,o in zip(good_el_l,good_mu_l,good_el_t,good_mu_t,good_el_m,good_mu_m,good_el_f,good_mu_f):
for i,j in zip(good_el,good_mu):
    plot_multicos(i,j,pol_list,where_save)
    print('+'*20)
    print(i,j)

print('plotting executed')
######################################################
plt.figure(1)
art_l = []
lgd_l = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1),ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_l.append(lgd_l)
#plt.title('Longitudinal polarization, '+reco_type)
ymin, ymax = plt.ylim()
plt.ylim((ymin,1.1*ymax))
plt.annotate(r'W$_{\mathbf{L}}$W$_{\mathbf{L}}$ polarization',xy=(-0.8, ymax),fontsize=14,weight='bold')
plt.xlabel(xlabel)
plt.ylabel('Number of events')


plt.figure(2)
art_t = []
lgd_t = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_t.append(lgd_t)
#plt.title('Transverse polarization, '+reco_type)
plt.xlabel(xlabel)
ymin, ymax = plt.ylim()
plt.ylim((ymin,1.1*ymax))
plt.annotate(r'W$_{\mathbf{T}}$W$_{\mathbf{T}}$ polarization',xy=(-0.8, ymax),fontsize=14,weight='bold')
plt.ylabel('Number of events')

plt.figure(3)
art_u = []
lgd_u = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_u.append(lgd_u)
#plt.title('Unpolarized OSP, '+reco_type)
plt.xlabel(config.get('plotting','xlabel'))
#plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')
ymin, ymax = plt.ylim()
plt.annotate(r'W$_{\mathbf{T}}$W$_{\mathbf{L}}$ polarization',xy=(-0.8, 1.1*ymax),fontsize=14,weight='bold')
plt.ylim((0, 1.2*plt.ylim()[1]))


plt.figure(4)
art_f = []
lgd_f = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_f.append(lgd_f)
#plt.title('Full computation, '+reco_type)
plt.xlabel(config.get('plotting','xlabel'))
#plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')
ymin, ymax = plt.ylim()
plt.annotate(r'Full computation',xy=(-0.8, ymax),fontsize=14,weight='bold')



fig_long.savefig(where_save + file_name + '_long.pdf', additional_artists=art_l,bbox_inches="tight")
fig_trans.savefig(where_save + file_name + '_trans.pdf', additional_artists=art_t,bbox_inches="tight")
fig_mix.savefig(where_save + '/theta_unpol.pdf', additional_artists=art_u,bbox_inches="tight")
fig_full.savefig(where_save + '/theta_full.pdf', additional_artists=art_f,bbox_inches="tight")

copyfile(args.config, where_save+ '/thisconfig.cfg')
print('figures saved into '+where_save)