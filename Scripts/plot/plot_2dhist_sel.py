#!/usr/bin/env python3
"""

  Michele Grossi <michele.grossi@it.ibm.com>
  Jakob Novak <janob.novak.cern.ch>
  Version 1.0, July 2020

  USAGE: python3  plot_2dhist_sel.py -c JobOption/NNplot_config.cfg
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
import matplotlib.colors as colors
from shutil import copyfile

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
hdf_unpol = pd.read_hdf(config.get('input','data-unpol'))
hdf_full_comp = pd.read_hdf(config.get('input','data-fulcomp'))

if config.get('selection','zero-delta-only') == '1':
    hdf_long = hdf_long.query("mu_delta==0")
    hdf_trans = hdf_trans.query("mu_delta==0")
    hdf_unpol = hdf_unpol.query("mu_delta==0")
    hdf_full_comp = hdf_full_comp.query("mu_delta==0")

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
    hdf_unpol.drop(bad_mod,axis=1)
    hdf_full_comp.drop(bad_mod,axis=1)
###

#################################################################plotting
binningx = config.get('plotting', 'binningx')
binningx = binningx.replace('\\', '')
binsx = binningx.split(',')
b1x,b2x,b3x = float(binsx[0]), float(binsx[1]), float(binsx[2])

binningy = config.get('plotting', 'binningy')
binningy = binningy.replace('\\', '')
binsy = binningy.split(',')
b1y,b2y,b3y = float(binsy[0]), float(binsy[1]), float(binsy[2])


good = []
for wildcard in config.get('selection','wildcard').split(','):
    # if config.get('selection','type') == 'binary': # These two lines need to be commented out
    #     wildcard += '_rounded_score'               # to train on autoai output or SELECTION CRITERIA
    # elif config.get('selection','type') == 'regression':
    #     wildcard += '_pred'
    good = good + fnmatch.filter(hdf_long.columns,wildcard)

if config.get('selection','type') == 'binary':

    variable0x = config.get('plotting','variable0x')
    variable1x = config.get('plotting','variable1x')
    variable0y = config.get('plotting','variable0y')
    variable1y = config.get('plotting','variable1y')
    variable0x = 'var0x = ' + variable0x
    variable1x = 'var1x = ' + variable1x
    variable0y = 'var0y = ' + variable0y
    variable1y = 'var1y = ' + variable1y

    hdf_long.eval(variable0x, inplace=True)
    hdf_long.eval(variable1x, inplace=True)
    hdf_trans.eval(variable0x, inplace=True)
    hdf_trans.eval(variable1x, inplace=True)
    hdf_unpol.eval(variable0x, inplace=True)
    hdf_unpol.eval(variable1x, inplace=True)
    hdf_full_comp.eval(variable0x, inplace=True)
    hdf_full_comp.eval(variable1x, inplace=True)

    hdf_long.eval(variable0y, inplace=True)
    hdf_long.eval(variable1y, inplace=True)
    hdf_trans.eval(variable0y, inplace=True)
    hdf_trans.eval(variable1y, inplace=True)
    hdf_unpol.eval(variable0y, inplace=True)
    hdf_unpol.eval(variable1y, inplace=True)
    hdf_full_comp.eval(variable0y, inplace=True)
    hdf_full_comp.eval(variable1y, inplace=True)

    s_lx = hdf_long[['var0x','var1x']].values
    s_ly = hdf_long[['var0y','var1y']].values

    s_tx = hdf_trans[['var0x','var1x']].values
    s_ty = hdf_trans[['var0y','var1y']].values

    s_ux = hdf_unpol[['var0x','var1x']].values
    s_uy = hdf_unpol[['var0y','var1y']].values

    s_fx = hdf_full_comp[['var0x','var1x']].values
    s_fy = hdf_full_comp[['var0y','var1y']].values
    

""""""
def plot_bin(name, avlb_pol, where, random=False):

    for pol_type in avlb_pol:

        pattern = config.get('legend','entry').split(':')
        entry = re.sub(pattern[0],pattern[1], name.rstrip())
        if random: entry = 'Random'

        if (config.get('plotting', 'normalize') == '1'):
            normalize = True
        else:
            normalize = False

        if pol_type == 'long':
            if random: score_l = np.random.randint(0,2,[s_lx.shape[0],])
            else: score_l = hdf_long[name]
            if (config.get('plotting', 'invert') == '1'):
                cos_lx = [s_lx[i, int(not bool(sign))] for i, sign in enumerate(score_l)]
                cos_ly = [s_ly[i, int(not bool(sign))] for i, sign in enumerate(score_l)]
            else:
                cos_lx = [s_lx[i, sign] for i, sign in enumerate(score_l)]
                cos_ly = [s_ly[i, sign] for i, sign in enumerate(score_l)]

            plt.figure(1)
            h2_long = plt.hist2d(cos_lx, cos_ly, [np.arange(b1x, b2x, b3x),np.arange(b1y, b2y, b3y)],  density=normalize, label = name, norm = colors.LogNorm())
            plt.colorbar(h2_long[3])
###########################fino qui ########################
        elif pol_type == 'trans':
            if random: score_t = np.random.randint(0,2,[s_tx.shape[0],])
            else: score_t = hdf_trans[name]
            if (config.get('plotting', 'invert') == '1'):
                cos_tx = [s_tx[i, int(not bool(sign))] for i, sign in enumerate(score_t)]
                cos_ty = [s_ty[i, int(not bool(sign))] for i, sign in enumerate(score_t)]
            else:
                cos_tx = [s_tx[i, sign] for i, sign in enumerate(score_t)]
                cos_ty = [s_ty[i, sign] for i, sign in enumerate(score_t)]

            plt.figure(2)
            h2_trans = plt.hist2d(cos_tx, cos_ty, [np.arange(b1x, b2x, b3x),np.arange(b1y, b2y, b3y)],  density=normalize, norm = colors.LogNorm()) 
            plt.colorbar(h2_trans[3])

        elif pol_type == 'unpol':
            if random: score_u = np.random.randint(0,2,[s_ux.shape[0],])
            else: score_u = hdf_unpol[name]
            if (config.get('plotting', 'invert') == '1'):
                cos_ux = [s_ux[i, int(not bool(sign))] for i, sign in enumerate(score_u)]
                cos_uy = [s_uy[i, int(not bool(sign))] for i, sign in enumerate(score_u)]
            else:
                cos_ux = [s_ux[i, sign] for i, sign in enumerate(score_u)]
                cos_uy = [s_uy[i, sign] for i, sign in enumerate(score_u)]

            plt.figure(3)
            h2_unpol = plt.hist2d(cos_ux, cos_uy, [np.arange(b1x, b2x, b3x),np.arange(b1y, b2y, b3y)],  density=normalize)

        elif pol_type == 'fullcomp':
            if random: score_f = np.random.randint(0,2,[s_fx.shape[0],])
            else: score_f = hdf_full_comp[name]
            if (config.get('plotting', 'invert') == '1'):
                cos_fx = [s_fx[i, int(not bool(sign))] for i, sign in enumerate(score_f)]
                cos_fy = [s_fy[i, int(not bool(sign))] for i, sign in enumerate(score_f)]
            else:
                cos_fx = [s_fx[i, sign] for i, sign in enumerate(score_f)]
                cos_fy = [s_fy[i, sign] for i, sign in enumerate(score_f)]

            plt.figure(4)
            h2_full = plt.hist2d(cos_fx, cos_fy, [np.arange(b1x, b2x, b3x),np.arange(b1y, b2y, b3y)],  density=normalize)

        else:
            print('wrong polarization')
    if random: name = 'random'
    np.savez(where + '/h_' + name, unpol=h2_unpol, trans=h2_trans, long=h2_long, fulcomp = h2_full)

""""""


if (config.get('plotting', 'normalize') == '1'):
    normalize = True
else:
    normalize = False

#here create the figure
#######LONGITUDINAL
fig_long = plt.figure(1)
# if (config.get('plotting', 'truth') == '1'):
#     h_long_true = plt.hist(hdf_long['truth_cos_theta'],np.arange(b1, b2, b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='Truth')

# # #########transverse
fig_trans = plt.figure(2)
# if (config.get('plotting', 'truth') == '1'):
#     h_trans_true = plt.hist(hdf_trans['truth_cos_theta'], np.arange(b1,b2,b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='Truth')

# #######unpolarized
# fig_unpol = plt.figure(3)
# if (config.get('plotting', 'truth') == '1'):
#     h_unpol_true = plt.hist(hdf_unpol['truth_cos_theta'], np.arange(b1,b2,b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='Truth')

# # ######full computation
# fig_full = plt.figure(4)
# if (config.get('plotting', 'truth') == '1'):
#     h_fullcomp_true = plt.hist(hdf_full_comp['truth_cos_theta'], np.arange(b1,b2,b3), histtype='stepfilled', facecolor='w', hatch='//', edgecolor='C0', density=normalize, linewidth=2, label='Truth')

# # ########################   saving all truth things
# if (config.get('plotting', 'truth') == '1'):
#     np.savez(where_save + '/h_truth',unpol=h_unpol_true, trans=h_trans_true, long=h_long_true, fulcomp = h_fullcomp_true)
# # ####################################

#############looping through selected model
print('looping through selected models:')

if config.get('plotting','random-choice') == '1':
    print('random')
    plot_bin('random',pol_list,where_save,True)    

for c in good:
    #here implement check if binary or regression!
    print(c)
    if config.get('selection','type') == 'binary':
        plot_bin(c,pol_list,where_save)
    else:
        raise ValueError('Error: wrong evaluation type selected')
print('plotting executed')

if config.get('selection','type') == 'binary':
    reco_type = 'classification'

plt.figure(1)
plt.xlabel(config.get('plotting','xlabel'))
plt.ylabel(config.get('plotting','ylabel'))

plt.figure(2)
plt.xlabel(config.get('plotting','xlabel'))
plt.ylabel(config.get('plotting','ylabel'))


fig_long.savefig(where_save + '/theta_pl_long.pdf')
fig_trans.savefig(where_save + '/theta_pl_trans.pdf')
#fig_unpol.savefig(where_save + '/theta_unpol.pdf', additional_artists=art_u,bbox_inches="tight")
#fig_full.savefig(where_save + '/theta_full.pdf', additional_artists=art_f,bbox_inches="tight")

copyfile(args.config, where_save+ '/thisconfig.cfg')
print('figures saved into '+where_save)
