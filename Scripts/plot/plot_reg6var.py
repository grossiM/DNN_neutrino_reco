#!/usr/bin/env python3
"""
  Michele Grossi <michele.grossi@it.ibm.com>
  Jakob Novak <janob.novak.cern.ch>
  Version 1.0, June 2020
  USAGE: python3  plot_ful_lep.py -c JobOption/***_config.cfg
  """
#non faccio stacked ma appendo i dataset

import os
import sys

repo = os.environ['NEW_REPO']
sys.path.append(repo + '/DNN_neutrino_reco/Utils/DataHandler')

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

if os.path.exists(where_save+'/theta_trans.pdf') or \
   os.path.exists(where_save+'/theta_long.pdf'):
    raise ValueError('plots in the '+where_save+' would be overwritten, exiting')
if not os.path.exists(where_save):
    os.system('mkdir ' + where_save)


###data reading & checking
####hdf5 reading, for time being only trans and long component are present
hdf_long = pd.read_hdf(config.get('input','data-long'))
hdf_trans = pd.read_hdf(config.get('input','data-trans'))
######
avlb_data = np.zeros((2, 1), dtype=bool) #put 4 instead of2 in case all 4 samples are present
try:
    hdf_long, hdf_trans#, hdf_unpol, hdf_full_comp
    avlb_data[0] = True
    avlb_data[1] = True
    pol_list = ['long','trans']
except NameError:
    print('not all polarized calibrated sample provided')


#################################################################selection and removal
#test here
to_rm = config.get('selection','discard').split(',')
print(' \n model discarded: {0} '.format(str(to_rm)))

###
for model_to_rm in to_rm:
    bad_mod = fnmatch.filter(hdf_long.columns, '*'+ model_to_rm + '*')
    #bad_mod = fnmatch.filter(hdf_long.columns, model_to_rm)
    print('discarded branches:')
    print(bad_mod)
    hdf_long = hdf_long.drop(bad_mod,axis=1)
    hdf_trans = hdf_trans.drop(bad_mod,axis=1)

###
#################################################################plotting
binning = config.get('plotting', 'binning')
binning = binning.replace('\\', '')
bins = binning.split(',')
b1,b2,b3 = float(bins[0]), float(bins[1]), float(bins[2])


good_mu_l_px = []
good_mu_l_py = []
good_mu_l_pz = []
good_el_l_px = []
good_el_l_py = []
good_el_l_pz = []
good_mu_t_px = []
good_mu_t_py = []
good_mu_t_pz = []
good_el_t_px = []
good_el_t_py = []
good_el_t_pz = []

cos2_long = {}
cos2_trans = {}
for wildcard in config.get('selection','wildcard').split(','):

    if config.get('selection','type') == 'reg6var':
        #map cat0= el_truth_cos_theta, cat1= mu_truth_cos_theta --> neu60hid2bat64_cat0_e100  neu60hid2bat64_cat1_e100
        lep_cat_list = ['cat0','cat1','cat2','cat3','cat4','cat5']
        ##########long
        #muon
        mu_cos_list_long_px = (fnmatch.filter(hdf_long.columns, '*'+lep_cat_list[3]+'*'))
        mu_cos_list_long_py = (fnmatch.filter(hdf_long.columns, '*'+lep_cat_list[4]+'*'))
        mu_cos_list_long_pz = (fnmatch.filter(hdf_long.columns, '*'+lep_cat_list[5]+'*'))
        #electron
        el_cos_list_long_px = (fnmatch.filter(hdf_long.columns, '*'+lep_cat_list[0]+'*'))
        el_cos_list_long_py = (fnmatch.filter(hdf_long.columns, '*'+lep_cat_list[1]+'*'))
        el_cos_list_long_pz = (fnmatch.filter(hdf_long.columns, '*'+lep_cat_list[2]+'*'))
        ##########trans
         #muon
        mu_cos_list_trans_px = (fnmatch.filter(hdf_trans.columns, '*'+lep_cat_list[3]+'*'))
        mu_cos_list_trans_py = (fnmatch.filter(hdf_trans.columns, '*'+lep_cat_list[4]+'*'))
        mu_cos_list_trans_pz = (fnmatch.filter(hdf_trans.columns, '*'+lep_cat_list[5]+'*'))
        #electron
        el_cos_list_trans_px = (fnmatch.filter(hdf_trans.columns, '*'+lep_cat_list[0]+'*'))
        el_cos_list_trans_py = (fnmatch.filter(hdf_trans.columns, '*'+lep_cat_list[1]+'*'))
        el_cos_list_trans_pz = (fnmatch.filter(hdf_trans.columns, '*'+lep_cat_list[2]+'*'))
    

        good_mu_l_px = good_mu_l_px + fnmatch.filter(mu_cos_list_long_px,wildcard + '*')
        good_mu_l_py = good_mu_l_py + fnmatch.filter(mu_cos_list_long_py,wildcard + '*')
        good_mu_l_pz = good_mu_l_pz + fnmatch.filter(mu_cos_list_long_pz,wildcard + '*')

        good_el_l_px = good_el_l_px + fnmatch.filter(el_cos_list_long_px,wildcard + '*')
        good_el_l_py = good_el_l_py + fnmatch.filter(el_cos_list_long_py,wildcard + '*')
        good_el_l_pz = good_el_l_pz + fnmatch.filter(el_cos_list_long_pz,wildcard + '*')

        good_mu_t_px = good_mu_t_px + fnmatch.filter(mu_cos_list_trans_px,wildcard + '*')
        good_mu_t_py = good_mu_t_py + fnmatch.filter(mu_cos_list_trans_py,wildcard + '*')
        good_mu_t_pz = good_mu_t_pz + fnmatch.filter(mu_cos_list_trans_pz,wildcard + '*')

        good_el_t_px = good_el_t_px + fnmatch.filter(el_cos_list_trans_px,wildcard + '*')
        good_el_t_py = good_el_t_py + fnmatch.filter(el_cos_list_trans_py,wildcard + '*')
        good_el_t_pz = good_el_t_pz + fnmatch.filter(el_cos_list_trans_pz,wildcard + '*')

    else:
        print('ciao bello')

""""""
def plot_multicos(name_el_l,name_mu_l,name_el_t,name_mu_t,avlb_pol, where):

    for pol_type in avlb_pol:

        pattern = config.get('legend','entry').split(':')
        entry = re.sub(pattern[0],pattern[1], name_el_l.rstrip())
        
        entry = re.sub('_cat0_e100', '', entry)
        entry = entry.split('_')[0].split('bat')[0]

        if (config.get('plotting', 'normalize') == '1'):
            normalize = True
        else:
            normalize = False

        if pol_type == 'long':
            plt.figure(1)
            plt.legend()
            cos2_long[name_el_l] = pd.DataFrame()
            cos2_long[name_el_l] = pd.concat([hdf_long[name_el_l],hdf_long[name_mu_l]])
            h_long = plt.hist(cos2_long[name_el_l].values, np.arange(b1, b2, b3), label=entry, density=normalize, histtype='step', linewidth=2)

        elif pol_type == 'trans':
            plt.figure(2)
            plt.legend()
            cos2_trans[name_el_t] = pd.DataFrame()
            cos2_trans[name_el_t] = pd.concat([hdf_trans[name_el_t],hdf_trans[name_mu_t]])
            h_trans = plt.hist(cos2_trans[name_el_t].values, np.arange(b1, b2, b3), label=entry, density=normalize, histtype='step', linewidth=2)

        else:
            print('wrong polarization')
    np.savez(where + '/h_' + entry, trans=h_trans, long=h_long)
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
    if config.get('selection','type') == 'reg6var':
        cos2_long['truth_cos_thetas'] = pd.concat([hdf_long['mu_truth_cos_theta'],hdf_long['el_truth_cos_theta']])
        h_long_true = plt.hist(cos2_long['truth_cos_thetas'],np.arange(b1, b2, b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='Truth Longitudinal')
    else: print('wrong selection type')   
# #########transverse
fig_trans = plt.figure(2)
if (config.get('plotting', 'truth') == '1'):
    if config.get('selection','type') == 'reg6var':
        cos2_trans['truth_cos_thetas'] = pd.concat([hdf_trans['mu_truth_cos_theta'],hdf_trans['el_truth_cos_theta']])
        h_trans_true = plt.hist(cos2_trans['truth_cos_thetas'],np.arange(b1, b2, b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='Truth Transverse')

# ########################   saving all truth things
if (config.get('plotting', 'truth') == '1'):
    np.savez(where_save + '/h_truth',trans=h_trans_true, long=h_long_true)
# ####################################

import handler_kinematics as hk

#############looping through selected model
print('looping through selected models:')
for elx,ely,elz,etx,ety,etz,mlx,mly,mlz,mtx,mty,mtz in \
                zip(good_el_l_px,good_el_l_py,good_el_l_pz, \
                    good_el_t_px,good_el_t_py,good_el_t_pz, \
                    good_mu_l_px,good_mu_l_py,good_mu_l_pz, \
                    good_mu_t_px,good_mu_t_py,good_mu_t_pz):
    print(elx[:-5])
    hdf_long[elx[:-5]+'_E'] = hdf_long.eval('sqrt('+elx+'**2+'+ely+'**2+'+elz+'**2)')
    hdf_trans[etx[:-5]+'_E'] = hdf_trans.eval('sqrt('+etx+'**2+'+ety+'**2+'+etz+'**2)')
    hdf_long[mlx[:-5]+'_E'] = hdf_long.eval('sqrt('+mlx+'**2+'+mly+'**2+'+mlz+'**2)')
    hdf_trans[mtx[:-5]+'_E'] = hdf_trans.eval('sqrt('+mtx+'**2+'+mty+'**2+'+mtz+'**2)')
    hdf_long[elx[:-5]+'_cos'] = hk.cos_theta(hdf_long,['el_px','el_py','el_pz','el_E',elx,ely,elz,elx[:-5]+'_E'])
    hdf_trans[etx[:-5]+'_cos'] = hk.cos_theta(hdf_trans,['el_px','el_py','el_pz','el_E',etx,ety,etz,etx[:-5]+'_E'])
    hdf_long[mlx[:-5]+'_cos'] = hk.cos_theta(hdf_long,['mu_px','mu_py','mu_pz','mu_E',mlx,mly,mlz,mlx[:-5]+'_E'])
    hdf_trans[mtx[:-5]+'_cos'] = hk.cos_theta(hdf_trans,['mu_px','mu_py','mu_pz','mu_E',mtx,mty,mtz,mtx[:-5]+'_E'])
    plot_multicos(elx[:-5]+'_cos',etx[:-5]+'_cos',mlx[:-5]+'_cos',mtx[:-5]+'_cos',pol_list,where_save)
        #raise ValueError('Error: wrong evaluation type selected')
print('plotting executed')
######################################################

plt.figure(1)
art_l = []
lgd_l = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1),ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_l.append(lgd_l)
#plt.title('Longitudinal polarization, '+reco_type)
plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')


plt.figure(2)
art_t = []
lgd_t = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_t.append(lgd_t)
#plt.title('Transverse polarization, '+reco_type)
plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')

fig_long.savefig(where_save + '/theta_long.pdf', additional_artists=art_l,bbox_inches="tight")
fig_trans.savefig(where_save + '/theta_trans.pdf', additional_artists=art_t,bbox_inches="tight")

long_orig = config.get('input','data-long').split('/')[-1]
base_long = long_orig.split('.')
base_long.insert(2,'boosted')
sample_long = '.'.join(base_long)

trans_orig = config.get('input','data-trans').split('/')[-1]
base_trans = trans_orig.split('.')
base_trans.insert(2,'boosted')
sample_trans = '.'.join(base_trans)

hdf_long.to_hdf(os.path.dirname(config.get('input','data-long')) + '/' + sample_long,'long',mode='w',format ='table')
hdf_trans.to_hdf(os.path.dirname(config.get('input','data-trans')) + '/' + sample_trans,'trans',mode='w',format ='table')

copyfile(args.config, where_save+ '/thisconfig.cfg')
print('figures saved into '+where_save)
print('evaluated boosted file saved into '+ os.path.dirname(config.get('input','data-trans')))