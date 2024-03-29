#!/usr/bin/env python3
"""

  Michele Grossi <michele.grossi@it.ibm.com>
  Jakob Novak <janob.novak.cern.ch>
  Version 1.0, February 2020

  USAGE: 
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

repo = os.environ['NEW_REPO']
sys.path.append(repo + '/DNN_neutrino_reco/DNN_main/Evaluation')

from sklearn.metrics import roc_auc_score, roc_curve
import optimizeThr as ot

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
args = parser.parse_args()

config = configparser.ConfigParser()
config.optionxform = str
config.read(args.config)

#this part should be implemented if the for cicle to change the folder name according to all selection list
where_save =  config.get('output','output-folder')

if os.path.exists(where_save+'/roc_full.pdf') or \
   os.path.exists(where_save+'/roc_unpol.pdf') or \
   os.path.exists(where_save+'/roc_trans.pdf') or \
   os.path.exists(where_save+'/roc_long.pdf'):
    raise ValueError('plots in the '+where_save+' would be overwritten, exiting')
if not os.path.exists(where_save):
    os.system('mkdir ' + where_save)


###data reading & checking
####hdf5 reading
hdf_long = pd.read_hdf(config.get('input','data-long'))
hdf_trans = pd.read_hdf(config.get('input','data-trans'))
hdf_unpol = pd.read_hdf(config.get('input','data-unpol'))
hdf_full_comp = pd.read_hdf(config.get('input','data-fulcomp'))

query = config.get('selection','filter')
if query != '':
    hdf_long = hdf_long.query(query)
    hdf_trans = hdf_trans.query(query)
    hdf_unpol = hdf_unpol.query(query)
    hdf_full_comp = hdf_full_comp.query(query)

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
    bad_mod = fnmatch.filter(hdf_long.columns, model_to_rm)

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
    good_all = fnmatch.filter(hdf_long.columns,wildcard+'*')
    good_rounded = fnmatch.filter(good_all,'*_rounded_score')
    good += set(good_all).difference(good_rounded)
    
""""""

def roundScore(score, thr):
    indeces = np.argwhere(score > thr)
    for index in indeces:
        score[index] = 1
    indeces = np.argwhere(score <= thr)
    for index in indeces:
        score[index] = 0
    score = score.astype(int)

    return score

""""""

def plot_model(name, avlb_pol, figNumber):

    for pol_type in avlb_pol:

        pattern = config.get('legend','entry').split(':')
        entry = re.sub(pattern[0],pattern[1], name.rstrip())
        thr = float(config.get('selection','working-point'))

        if pol_type == 'long':

            plt.figure(1)
            auc = roc_auc_score(hdf_long['v_mu_label'].values, hdf_long[name].values)
            print('>>> Longitudinal polarization')
            print(">>> AUC: ",auc)
            fp , tp, th = roc_curve(hdf_long['v_mu_label'].values, hdf_long[name].values)
            if thr < 0: thr, _, _ = ot.optimizeThr(fp,tp,th)
            plt.plot(fp, tp, label=entry)

            selection = roundScore(hdf_long[name].values, thr)
            nall = selection.shape[0]
            comparison = np.ones((nall,), dtype=bool)
            np.equal(hdf_long['v_mu_label'].values,selection,comparison)
            print(">>> Fraction of correct predictions: "+str(np.sum(comparison)/nall))

            plt.figure(figNumber)
            plt.plot(fp, tp, label='Long.')

        elif pol_type == 'trans':
            
            plt.figure(2)
            auc = roc_auc_score(hdf_trans['v_mu_label'].values, hdf_trans[name].values)
            print('>>> Transverse polarization')
            print(">>> AUC: ",auc)
            fp , tp, th = roc_curve(hdf_trans['v_mu_label'].values, hdf_trans[name].values)
            if thr < 0: thr, _, _ = ot.optimizeThr(fp,tp,th)
            plt.plot(fp, tp, label=entry)

            selection = roundScore(hdf_trans[name].values, thr)
            nall = selection.shape[0]
            comparison = np.ones((nall,), dtype=bool)
            np.equal(hdf_trans['v_mu_label'].values,selection,comparison)
            print(">>> Fraction of correct predictions: "+str(np.sum(comparison)/nall))

            plt.figure(figNumber)
            plt.plot(fp, tp, label='Trans.')

        elif pol_type == 'unpol':
            
            plt.figure(3)
            auc = roc_auc_score(hdf_unpol['v_mu_label'].values, hdf_unpol[name].values)
            print('>>> Unpolarized OSP')
            print(">>> AUC: ",auc)
            fp , tp, th = roc_curve(hdf_unpol['v_mu_label'].values, hdf_unpol[name].values)
            if thr < 0: thr, _, _ = ot.optimizeThr(fp,tp,th)
            plt.plot(fp, tp, label=entry)

            selection = roundScore(hdf_unpol[name].values, thr)
            nall = selection.shape[0]
            comparison = np.ones((nall,), dtype=bool)
            np.equal(hdf_unpol['v_mu_label'].values,selection,comparison)
            print(">>> Fraction of correct predictions: "+str(np.sum(comparison)/nall))

            plt.figure(figNumber)
            plt.plot(fp, tp, label='Un. OSP')

        elif pol_type == 'fullcomp':
            
            plt.figure(4)
            auc = roc_auc_score(hdf_full_comp['v_mu_label'].values, hdf_full_comp[name].values)
            print('>>> Full computation')
            print(">>> AUC: ",auc)
            fp , tp, th = roc_curve(hdf_full_comp['v_mu_label'].values, hdf_full_comp[name].values)
            if thr < 0: thr, _, _ = ot.optimizeThr(fp,tp,th)
            plt.plot(fp, tp, label=entry)

            selection = roundScore(hdf_full_comp[name].values, thr)
            nall = selection.shape[0]
            comparison = np.ones((nall,), dtype=bool)
            np.equal(hdf_full_comp['v_mu_label'].values,selection,comparison)
            print(">>> Fraction of correct predictions: "+str(np.sum(comparison)/nall))

            plt.figure(figNumber)
            plt.plot(fp, tp, label='Full comp.')

        else:
            print('wrong polarization')


# ####################################

#############looping through selected model
print('looping through selected models:')

for figNumber, c in enumerate(good):
    #here implement check if binary or regression! o sopra
    print('\n\n\n\n')
    print('>>> Model '+c+':')

    plot_model(c,pol_list,figNumber+5)
    fig_model = plt.figure(figNumber+5)
    pattern = config.get('plotting','model-title').split(':')
    title = re.sub(pattern[0],pattern[1], c.rstrip())
    # plt.title(title)
    plt.legend(loc='upper left', ncol=2, fancybox=True, fontsize=int(config.get('legend','fontsize')))
    plt.xlabel('Contamination')
    plt.ylabel('Efficiency')
    fig_model.savefig(where_save+'/roc_'+c+'.pdf')

print('\n\n\n\n')
print('plotting executed')

#here create the figure
#######longitudinal
fig_long = plt.figure(1)
# plt.title('Longitudinal polarization - ROC curves')
plt.legend(loc='upper left', ncol=2, fancybox=True, fontsize=int(config.get('legend','fontsize')))
plt.xlabel('Contamination')
plt.ylabel('Efficiency')

# #########transverse
fig_trans = plt.figure(2)
# plt.title('Transverse polarization - ROC curves')
plt.legend(loc='upper left', ncol=2, fancybox=True, fontsize=int(config.get('legend','fontsize')))
plt.xlabel('Contamination')
plt.ylabel('Efficiency')

# #######unpolarized
fig_unpol = plt.figure(3)
# plt.title('Unpolarized OSP - ROC curves')
plt.legend(loc='upper left', ncol=2, fancybox=True, fontsize=int(config.get('legend','fontsize')))
plt.xlabel('Contamination')
plt.ylabel('Efficiency')

# ######full computation
fig_full = plt.figure(4)
# plt.title('Full computation - ROC curves')
plt.legend(loc='upper left', ncol=2, fancybox=True, fontsize=int(config.get('legend','fontsize')))
plt.xlabel('Contamination')
plt.ylabel('Efficiency')

fig_long.savefig(where_save + '/roc_long.pdf')
fig_trans.savefig(where_save + '/roc_trans.pdf')
fig_unpol.savefig(where_save + '/roc_unpol.pdf')
fig_full.savefig(where_save + '/roc_full.pdf')

print('figures saved into '+where_save)
