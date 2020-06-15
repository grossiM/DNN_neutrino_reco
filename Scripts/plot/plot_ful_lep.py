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
long_rm = hdf_long.columns[hdf_long.isna().any()].tolist()
print('Models containing Nan: \n {0}'.format(long_rm))
trans_rm = hdf_trans.columns[hdf_trans.isna().any()].tolist()

hdf_long = hdf_long.drop(long_rm,axis=1)
hdf_trans = hdf_trans.drop(trans_rm,axis=1)

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


good_mu_l = []
good_el_l = []
good_mu_t = []
good_el_t = []

good_mu_pxl = []
good_mu_pyl = []
good_mu_pxt = []
good_mu_pyt = []

good_el_pxl = []
good_el_pyl = []
good_el_pxt = []
good_el_pyt = []

cos2_long = {}
cos2_trans = {}

ptt_long = {}
ptt_trans = {}

for wildcard in config.get('selection','wildcard').split(','):

    if config.get('selection','type') == 'regcostheta':
        print(' Plotting regression on costheta')
        #map cat0= el_truth_cos_theta, cat1= mu_truth_cos_theta --> neu60hid2bat64_cat0_e100  neu60hid2bat64_cat1_e100
        lep_cat_list = ['cat0','cat1']
        ##########long
        #muon
        mu_cos_list_long = (fnmatch.filter(hdf_long.columns, '*'+lep_cat_list[1]+'*'))
        #electron
        el_cos_list_long = (fnmatch.filter(hdf_long.columns, '*'+lep_cat_list[0]+'*'))
        ##########trans
         #muon
        mu_cos_list_trans = (fnmatch.filter(hdf_trans.columns, '*'+lep_cat_list[1]+'*'))
        #electron
        el_cos_list_trans = (fnmatch.filter(hdf_trans.columns, '*'+lep_cat_list[0]+'*'))
    

        good_mu_l = good_mu_l + fnmatch.filter(mu_cos_list_long,wildcard + '*')
        good_el_l = good_el_l + fnmatch.filter(el_cos_list_long,wildcard + '*')
        good_mu_t = good_mu_t + fnmatch.filter(mu_cos_list_trans,wildcard + '*')
        good_el_t = good_el_t + fnmatch.filter(el_cos_list_trans,wildcard + '*')
        
    elif config.get('selection','type') == 'regneutrinos':
        print(' Plotting regression on total pt of neutrinos')
        #mapping from training config: mapping={'cat0': 'v_mu_px', 'cat1': 'v_mu_py','cat2': 'v_mu_pz','cat3': 'v_el_px', 'cat4': 'v_el_py', 'cat5': 'v_el_pz'}
        lep_cat_list = ['cat0','cat1','cat3', 'cat4']
        ##########long
        #muon
        mu_px_list_long = (fnmatch.filter(hdf_long.columns, '*'+lep_cat_list[0]+'*'))
        mu_py_list_long = (fnmatch.filter(hdf_long.columns, '*'+lep_cat_list[1]+'*'))
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


        good_mu_pxl = good_mu_pxl + fnmatch.filter(mu_px_list_long,wildcard + '*')
        good_mu_pyl = good_mu_pyl + fnmatch.filter(mu_py_list_long,wildcard + '*')
        good_mu_pxt = good_mu_pxt + fnmatch.filter(mu_px_list_trans,wildcard + '*')
        good_mu_pyt = good_mu_pyt + fnmatch.filter(mu_py_list_trans,wildcard + '*')

        good_el_pxl = good_el_pxl + fnmatch.filter(el_px_list_long,wildcard + '*')
        good_el_pyl = good_el_pyl + fnmatch.filter(el_py_list_long,wildcard + '*')
        good_el_pxt = good_el_pxt + fnmatch.filter(el_px_list_trans,wildcard + '*')
        good_el_pyt = good_el_pyt + fnmatch.filter(el_py_list_trans,wildcard + '*')
    else:
        print('wrong selection type')

""""""
def plot_multicos(name_el_l,name_mu_l,name_el_t,name_mu_t,avlb_pol, where):

    for pol_type in avlb_pol:

        pattern = config.get('legend','entry').split(':')
        entry = re.sub(pattern[0],pattern[1], name_el_l.rstrip())
        
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
            cos2_long[name_el_l] = pd.concat([hdf_long[name_el_l],hdf_long[name_mu_l]])
            
            h_long = plt.hist(cos2_long[name_el_l].values, np.arange(b1, b2, b3), label=entry, density=normalize, histtype='step', linewidth=2)

        elif pol_type == 'trans':
            plt.figure(2)
            plt.legend()
            #cos2_trans[name_el_t] = pd.DataFrame()
            cos2_trans[name_el_t] = pd.concat([hdf_trans[name_el_t],hdf_trans[name_mu_t]])

            h_trans = plt.hist(cos2_trans[name_el_t].values, np.arange(b1, b2, b3), label=entry, density=normalize, histtype='step', linewidth=2)

        else:
            print('wrong polarization')
    np.savez(where + '/h_' + entry, trans=h_trans, long=h_long)

""""""
""""""
def plot_neutrinospt(name_el_lx,name_el_ly,name_mu_lx,name_mu_ly,name_el_tx,name_el_ty,name_mu_tx,name_mu_ty,avlb_pol, where):

    for pol_type in avlb_pol:

        pattern = config.get('legend','entry').split(':')
        entry = re.sub(pattern[0],pattern[1], name_el_lx.rstrip())
        
        entry = re.sub('_cat0_pred', '', entry)
        entry = re.sub('_cat0_e100','', entry)
        entry = re.sub('_cat3_e100','', entry)
        entry = entry.split('bat')[0]

        if (config.get('plotting', 'normalize') == '1'):
            normalize = True
        else:
            normalize = False

        if pol_type == 'long':
            plt.figure(1)
            plt.legend()
            hdf_long[name_el_lx[:-10]+'_v_el_pt'] = np.sqrt(hdf_long[name_el_lx]**2 + hdf_long[name_el_ly]**2)
            hdf_long[name_mu_lx[:-10]+'_v_mu_pt'] = np.sqrt(hdf_long[name_mu_lx]**2 + hdf_long[name_mu_ly]**2)
            ptt_long[entry+'_ptvv'] = pd.DataFrame()
            ptt_long[entry+'_ptvv'] = pd.concat([hdf_long[name_el_lx[:-10]+'_v_el_pt'],hdf_long[name_mu_lx[:-10]+'_v_mu_pt']])
        
            h_long = plt.hist(ptt_long[entry+'_ptvv'].values, bins = 100, label=entry, density=normalize, histtype='step', linewidth=2)

        elif pol_type == 'trans':
            plt.figure(2)
            plt.legend()
            hdf_trans[name_el_tx[:-10]+'_v_el_pt'] = np.sqrt(hdf_trans[name_el_tx]**2 + hdf_trans[name_el_ty]**2)
            hdf_trans[name_mu_tx[:-10]+'_v_mu_pt'] = np.sqrt(hdf_trans[name_mu_tx]**2 + hdf_trans[name_mu_ty]**2)
            ptt_trans[entry+'_ptvv'] = pd.DataFrame()
            ptt_trans[entry+'_ptvv'] = pd.concat([hdf_trans[name_el_tx[:-10]+'_v_el_pt'],hdf_trans[name_mu_tx[:-10]+'_v_mu_pt']])
            # print('T'*120)
            # print('ptt_trans')
            # print(ptt_trans[entry+'_ptvv'][:15])
            # print('*'*120)
            # print('*'*120)
            # print('*'*120)
            h_trans = plt.hist(ptt_trans[entry+'_ptvv'].values, bins = 100, label=entry, density=normalize, histtype='step', linewidth=2)

        else:
            print('wrong polarization')
    np.savez(where + '/h_' + entry, trans=h_trans, long=h_long)
""""""

if (config.get('plotting', 'normalize') == '1'):
    normalize = True
else:
    normalize = False

#here create the figure
#######LONGITUDINAL
fig_long = plt.figure(1)
if (config.get('plotting', 'truth') == '1'):
    if config.get('selection','type') == 'regcostheta':
        cos2_long['truth_cos_thetas'] = pd.concat([hdf_long['mu_truth_cos_theta'],hdf_long['el_truth_cos_theta']])
        h_long_true = plt.hist(cos2_long['truth_cos_thetas'],np.arange(b1, b2, b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='Truth Longitudinal')
    
    elif config.get('selection','type') == 'regneutrinos':
        ptt_long['pt_vv_truth'] = pd.concat([hdf_long['v_mu_pt'],hdf_long['v_el_pt']])
        h_long_true = plt.hist(ptt_long['pt_vv_truth'],bins = 100, histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='Truth Longitudinal')
        #pt_inf = min(hdf_long['pt_vv'])
        #pt_max = min(hdf_long['pt_vv'])

    else: print('wrong selection type')   
# #########TRANSVERSE
fig_trans = plt.figure(2)
if (config.get('plotting', 'truth') == '1'):
    if config.get('selection','type') == 'regcostheta':
        cos2_trans['truth_cos_thetas'] = pd.concat([hdf_trans['mu_truth_cos_theta'],hdf_trans['el_truth_cos_theta']])
        h_trans_true = plt.hist(cos2_trans['truth_cos_thetas'],np.arange(b1, b2, b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='Truth Transverse')
    
    elif config.get('selection','type') == 'regneutrinos':
        ptt_trans['pt_vv_truth'] = pd.concat([hdf_trans['v_mu_pt'],hdf_trans['v_el_pt']])
        h_trans_true = plt.hist(ptt_trans['pt_vv_truth'],bins = 100, histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='Truth Transverse')
        print(ptt_trans['pt_vv_truth'][:15])

    else: print('wrong selection type')    
# ########################   saving all truth things
if (config.get('plotting', 'truth') == '1'):
    np.savez(where_save + '/h_truth',trans=h_trans_true, long=h_long_true)
# ####################################

#############looping through selected model
print('looping through selected models:')
if config.get('selection','type') == 'regcostheta':
    xlabel = 'cos'+r'$\theta$'
    file_name = '/theta'
    for i,j,p,q in zip(good_el_l,good_mu_l,good_el_t,good_mu_t):
        plot_multicos(i,j,p,q,pol_list,where_save)
        print(i)

elif config.get('selection','type') == 'regneutrinos':
    xlabel = 'pt'+r'$_{\nu\nu}$'
    file_name = '/pt_vv'
    for i,j,k,l,p,q,r,s in zip(good_el_pxl,good_el_pyl,good_mu_pxl,good_mu_pyl,good_el_pxt,good_el_pyt,good_mu_pxt,good_mu_pyt):
        plot_neutrinospt(i,j,k,l,p,q,r,s,pol_list,where_save)
        print(i)
else:
    raise ValueError('Error: wrong evaluation type selected')

print('plotting executed')
######################################################
plt.figure(1)
art_l = []
lgd_l = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1),ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_l.append(lgd_l)
#plt.title('Longitudinal polarization, '+reco_type)
plt.xlabel(xlabel)
plt.ylabel('Number of events')


plt.figure(2)
art_t = []
lgd_t = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(config.get('legend','ncol')), fancybox=True, fontsize=int(config.get('legend','fontsize')))
art_t.append(lgd_t)
#plt.title('Transverse polarization, '+reco_type)
plt.xlabel(xlabel)
plt.ylabel('Number of events')

fig_long.savefig(where_save + file_name + '_long.pdf', additional_artists=art_l,bbox_inches="tight")
fig_trans.savefig(where_save + file_name + '_trans.pdf', additional_artists=art_t,bbox_inches="tight")

copyfile(args.config, where_save+ '/thisconfig.cfg')
print('figures saved into '+where_save)