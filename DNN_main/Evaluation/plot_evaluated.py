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

#plot_selection = []
#plot_selection.append(config.get('selection','pattern'))
print('selection ')

neu_sel = config.get('selection','neu-sel')
hid_sel = config.get('selection','hid-sel')
bat_sel = config.get('selection','bat-sel')

#neu20hid10bat8_e100

plot_selection = neu_sel +'_'+ hid_sel + '_' + bat_sel
print(plot_selection)
#this part should be implemented if the for cicle to change the folder name according to all selection list
where_save =  config.get('output','output-folder') + plot_selection

print(where_save)

if os.path.exists(where_save):
    raise ValueError('Error: folder '+ where_save +' already exists')
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
    print(pol_list)
except NameError:
    print('not all polarized calibrated sample provided')


#selection and removal
to_rm = config.get('selection','discard').split(',')
print('selection to remove : ' )
###
for r in range(len(to_rm)):
    print(to_rm[r])
    bad_mod = fnmatch.filter(hdf_long.columns, '*'+to_rm[r]+'*')

    hdf_long = hdf_long.drop(bad_mod,axis=1)
    hdf_trans.drop(bad_mod,axis=1)
    hdf_unpol.drop(bad_mod,axis=1)
    hdf_full_comp.drop(bad_mod,axis=1)
    print(bad_mod)
###
print(to_rm)

#plotting
binning = config.get('plotting', 'binning')
binning = binning.replace('\\', '')
bins = binning.split(',')
b1,b2,b3 = float(bins[0]), float(bins[1]), float(bins[2])


good = fnmatch.filter(hdf_long.columns,'*' + neu_sel+'*'+hid_sel+'*'+bat_sel+'*' )

if config.get('selection','type') == 'binary':

    s_l = hdf_long[['sol0_cos_theta','sol1_cos_theta']].values
    s_t = hdf_trans[['sol0_cos_theta','sol1_cos_theta']].values
    s_u = hdf_unpol[['sol0_cos_theta','sol1_cos_theta']].values
    s_f = hdf_full_comp[['sol0_cos_theta','sol1_cos_theta']].values
    
    rm_round_score = fnmatch.filter(hdf_long.columns,'*' + '_rounded_score' +'*')
    good.remove(rm_round_score[0])

""""""
def plot_bin(model, avlb_pol, where):
    name = model[:-5]
    for pol_type in avlb_pol:
        
        #score = hdf_long[model+'_rounded_score']
        # cos_l = [sl[i, sign[0]] for i, sign in enumerate(score)]
        # cos_t = [st[i, sign[0]] for i, sign in enumerate(score)]
        # cos_u = [su[i, sign[0]] for i, sign in enumerate(score)]
        # cos_f = [sf[i, sign[0]] for i, sign in enumerate(score)]

        if pol_type == 'long':
            score_l = hdf_long[name+'_rounded_score']
            cos_l = [s_l[i, sign] for i, sign in enumerate(score_l)]
            plt.figure(1)
            plt.legend()
            h_long = plt.hist(cos_l, np.arange(b1, b2, b3),alpha=0.3, label=name)

        elif pol_type == 'trans':
            score_t = hdf_trans[name+'_rounded_score']
            cos_t = [s_t[i, sign] for i, sign in enumerate(score_t)]
            plt.figure(2)
            plt.legend()
            h_trans = plt.hist(cos_t, np.arange(b1, b2, b3),alpha=0.3, label=name)

        elif pol_type == 'unpol':
            score_u = hdf_unpol[name+'_rounded_score']
            cos_u = [s_u[i, sign] for i, sign in enumerate(score_u)]
            plt.figure(3)
            plt.legend()
            h_unpol = plt.hist(cos_u, np.arange(b1, b2, b3),alpha=0.3, label=name)

        elif pol_type == 'fullcomp':
            score_f = hdf_full_comp[name+'_rounded_score']
            cos_f = [s_f[i, sign] for i, sign in enumerate(score_f)]
            plt.figure(4)
            plt.legend()
            h_full = plt.hist(cos_f, np.arange(b1, b2, b3),alpha=0.3, label=name)

        else:
            print('wrong polarization')
    np.savez(where + '/h_' + name, unpol=h_unpol, trans=h_trans, long=h_long, fulcomp = h_full)

""""""

""""""
def plot_reg(model,avlb_pol, where):
    name = model[:-5]
    for pol_type in avlb_pol:

        if pol_type == 'long':
            plt.figure(1)
            plt.legend()
            h_long = plt.hist(hdf_long[model].values, np.arange(b1, b2, b3),alpha=0.3, label=name)

        elif pol_type == 'trans':
            plt.figure(2)
            plt.legend()
            h_trans = plt.hist(hdf_trans[model].values, np.arange(b1, b2, b3),alpha=0.3, label=name)

        elif pol_type == 'unpol':
            plt.figure(3)
            plt.legend()
            h_unpol = plt.hist(hdf_unpol[model].values, np.arange(b1, b2, b3),alpha=0.3, label=name)

        elif pol_type == 'fullcomp':
            plt.figure(4)
            plt.legend()
            h_full = plt.hist(hdf_full_comp[model].values, np.arange(b1, b2, b3),alpha=0.3, label=name)

        else:
            print('wrong polarization')
    np.savez(where + '/h_' + name, unpol=h_unpol, trans=h_trans, long=h_long, fulcomp = h_full)
    #np.savez(self.config.get('evaluation', 'output') + '/h_' + model_dir, unpol=h_unpol, trans=h_trans, long=h_long, dlong = d_long, dtrans = d_trans)
""""""
#here create the figure
#######LONGITUDINAL
fig_long = plt.figure(1)
h_long_true = plt.hist(hdf_long['truth_cos_theta'],np.arange(b1, b2, b3),alpha = 0.5, edgecolor='black', linewidth=2.1, label='truth')

plt.figure(1)
plt.legend(loc='upper left', ncol=3, fancybox=True, fontsize='small')
plt.title('Longitudinal polarization')
plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')

# #########transverse
fig_trans = plt.figure(2)
h_trans_true = plt.hist(hdf_trans['truth_cos_theta'], np.arange(b1,b2,b3), alpha = 0.5, edgecolor='black', linewidth=2.1, label='truth')

plt.figure(2)
plt.legend(loc='upper left', ncol=3, fancybox=True, fontsize='small')
plt.title('Transverse polarization')
plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')

# #######unpolarized
fig_unpol = plt.figure(3)
h_unpol_true = plt.hist(hdf_unpol['truth_cos_theta'], np.arange(b1,b2,b3), alpha = 0.5, edgecolor='black', linewidth=2.1, label='truth')

plt.figure(3)
plt.legend(loc='upper left', ncol=3, fancybox=True, fontsize='small')
plt.title('Unpolarized polarization')
plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')

# ######full computation
fig_full = plt.figure(4)
h_fullcomp_true = plt.hist(hdf_full_comp['truth_cos_theta'], np.arange(b1,b2,b3), alpha = 0.5, edgecolor='black', linewidth=2.1, label='truth')


plt.figure(4)
plt.legend(loc='upper left', ncol=3, fancybox=True, fontsize='small')
plt.title('Full computation polarization')
plt.xlabel('cos'+r'$\theta$')
plt.ylabel('Number of events')


# ########################   saving all truth things
np.savez(where_save + '/h_truth',unpol=h_unpol_true, trans=h_trans_true, long=h_long_true, fulcomp = h_fullcomp_true)
# ####################################

#good = fnmatch.filter(hdf_long.columns, '*'+plot_selection[s]+'*')

#############looping through selected model
print('looping through selected model:')
print(good)
for c in good:
    #here implement check if binary or regression! o sopra
    #print(name)
    if config.get('selection','type') == 'binary':
        plot_bin(c,pol_list,where_save)
    elif config.get('selection','type') == 'regression':
        plot_reg(c,pol_list,where_save)
    else:
        raise ValueError('Error: wrong evaluation type selected')
print('plotting executed')
fig_long.savefig(where_save + '/theta_long.pdf')
fig_trans.savefig(where_save + '/theta_trans.pdf')
fig_unpol.savefig(where_save + '/theta_unpol.pdf')
fig_full.savefig(where_save + '/theta_full.pdf')

print('figures saved')
