#!/usr/bin/env python3
"""

  Michele Grossi <michele.grossi@it.ibm.com>
  Jakob Novak <janob.novak.cern.ch>
  Version 1.0, February 2020

  USAGE: python3  plot_scatter.py -c JobOption/NNplot_config.cfg
  """

import os
import sys
import configparser
import re
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

#this part should be implemented if the for cicle to change the folder name according to all selection list
where_save =  config.get('output','output-folder')

if os.path.exists(where_save+'/scatter_full.pdf') or \
   os.path.exists(where_save+'/scatter_unpol.pdf') or \
   os.path.exists(where_save+'/scatter_trans.pdf') or \
   os.path.exists(where_save+'/scatter_long.pdf'):
    raise ValueError('plots in the '+where_save+' would be overwritten, exiting')
if not os.path.exists(where_save):
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
except NameError:
    print('not all polarized calibrated sample provided')

hdf_long = hdf_long.reindex(sorted(hdf_long.columns), axis=1)
hdf_trans = hdf_trans.reindex(sorted(hdf_trans.columns), axis=1)
hdf_unpol = hdf_unpol.reindex(sorted(hdf_unpol.columns), axis=1)
hdf_full_comp = hdf_full_comp.reindex(sorted(hdf_full_comp.columns), axis=1)
print('hdf_long: ',hdf_long.columns)
#selection and removal
to_rm = config.get('selection','discard').split(',')
print(to_rm)

###
for model_to_rm in to_rm:
    bad_mod = fnmatch.filter(hdf_long.columns, '*'+ model_to_rm +'*')
    print('bad_mod')
    print(bad_mod)
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
    if config.get('selection','type') == 'binary':
        wildcard += '_rounded_score'
    good = good + fnmatch.filter(hdf_long.columns,wildcard)

if config.get('selection','type') == 'binary':

    s_l = hdf_long[['sol0_cos_theta','sol1_cos_theta']].values
    s_t = hdf_trans[['sol0_cos_theta','sol1_cos_theta']].values
    s_u = hdf_unpol[['sol0_cos_theta','sol1_cos_theta']].values
    s_f = hdf_full_comp[['sol0_cos_theta','sol1_cos_theta']].values
    
""""""
def plot_scat(name,avlb_pol):
    
    hid = name.split('bat')[0].split('hid')[1]
    neu = name.split('bat')[0].split('hid')[0].split('neu')[1]
    #entry = '{0} neu {1} hid. layers.'.format(neu,hid)
    entry = '{0}hid. layers.'.format(hid)
    

    for pol_type in avlb_pol:
        if pol_type == 'long':
            plt.figure(1)
            plt.legend()
    
            xl = hdf_long['truth_cos_theta'][0::777].values
            yl = hdf_long[name][0::777].values 
            plt.scatter(xl,yl,label=entry)
            plt.plot(xl,xl,c='black')

        elif pol_type == 'trans':
            plt.figure(2)
            plt.legend()
            
            xt = hdf_trans['truth_cos_theta'][0::777].values
            yt = hdf_trans[name][0::777].values 
            plt.scatter(xt,yt,label=entry)
            plt.plot(xt,xt,c='black')

        elif pol_type == 'unpol':
            plt.figure(3)
            plt.legend()
           
            xu = hdf_unpol['truth_cos_theta'][0::777].values
            yu = hdf_unpol[name][0::777].values 
            plt.scatter(xu,yu,label=entry)
            plt.plot(xu,xu,c='black')

        elif pol_type == 'fullcomp':
            plt.figure(4)
            plt.legend()
            
            xf = hdf_full_comp['truth_cos_theta'][0::777].values
            yf = hdf_full_comp[name][0::777].values 
            plt.scatter(xf,yf,label=entry)
            plt.plot(xf,xf,c='black')

        else:
            print('wrong polarization')
""""""


""""""
def scatt_diff(name,avlb_pol):

    pattern = config.get('legend','entry').split(':')
    entry = re.sub(pattern[0],pattern[1], name.rstrip())
    
    entry = re.sub('_e100', '', entry)



    for pol_type in avlb_pol:
        if pol_type == 'long':
            
            plt.figure(5)
            plt.legend()
            
            xdl = hdf_long['truth_cos_theta'][0::777].values
            ydl = hdf_long[name][0::777].values
            dl = ydl - xdl
            plt.scatter(xdl,dl,label=entry)
            
        elif pol_type == 'trans':
            
            plt.figure(6)
            plt.legend()
            
            xdt = hdf_trans['truth_cos_theta'][0::777].values
            ydt = hdf_trans[name][0::777].values
            dt = ydt - xdt
            plt.scatter(xdt,dt,label=entry)
            
            
        elif pol_type == 'unpol':
            
            plt.figure(7)
            plt.legend()
            
            xdu = hdf_unpol['truth_cos_theta'][0::777].values
            ydu = hdf_unpol[name][0::777].values
            du = ydu - xdu
            plt.scatter(xdu,du,label=entry)
            

        elif pol_type == 'fullcomp':
            
            plt.figure(8)
            plt.legend()
            
            xdf = hdf_full_comp['truth_cos_theta'][0::777].values
            ydf = hdf_full_comp[name][0::777].values
            df = ydf - xdf
            plt.scatter(xdf,df,label=entry)
        
        else:
            print('wrong polarization')
""""""

# ####################################

#############looping through selected model
print('looping through selected models:')

for c in good:
    
    print('\n\n\n\n')
    print('>>> Model '+c+':')
    plot_scat(c,pol_list)
    scatt_diff(c,pol_list)
print('\n\n\n\n')
print('plotting executed')

######################################here create the figure
#######longitudinal
fig_long = plt.figure(1)
#plt.title('Longitudinal ')
art_l = []
lgd_l = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2, fancybox=True, fontsize='small')
art_l.append(lgd_l)
ymin, ymax = plt.ylim()
plt.ylim((ymin,1.1*ymax))
plt.annotate(r'W$_{\mathbf{L}}$ polarization',xy=(-0.8, ymax),fontsize=14,weight='bold')

plt.xlabel('truth cos'+r'$\theta$')
plt.ylabel('reconstructed cos'+r'$\theta$')

# #########transverse
fig_trans = plt.figure(2)
art_t = []
lgd_t = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2, fancybox=True, fontsize='small')
art_t.append(lgd_t)
ymin, ymax = plt.ylim()
plt.ylim((ymin,1.1*ymax))
plt.annotate(r'W$_{\mathbf{T}}$ polarization',xy=(-0.8, ymax),fontsize=14,weight='bold')
#plt.title('Transverse')

plt.xlabel('truth cos'+r'$\theta$')
plt.ylabel('reconstructed cos'+r'$\theta$')

# #######unpolarized
fig_unpol = plt.figure(3)
art_u = []
lgd_u = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2, fancybox=True, fontsize='small')
art_u.append(lgd_u)
ymin, ymax = plt.ylim()
plt.ylim((ymin,1.1*ymax))
#plt.title('Unpolarized')
plt.annotate(r'W unpolarized',xy=(-0.8, ymax),fontsize=14,weight='bold')
plt.xlabel('truth cos'+r'$\theta$')
plt.ylabel('reconstructed cos'+r'$\theta$')

# ######full computation
fig_full = plt.figure(4)
art_f = []
lgd_f = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2, fancybox=True, fontsize='small')
art_f.append(lgd_f)
#plt.title('Full computation')
plt.xlabel('truth cos'+r'$\theta$')
plt.ylabel('reconstructed cos'+r'$\theta$')

fig_long.savefig(where_save + '/scatter_long.pdf',additional_artists=art_l,bbox_inches="tight")
fig_trans.savefig(where_save + '/scatter_trans.pdf',additional_artists=art_t,bbox_inches="tight")
fig_unpol.savefig(where_save + '/scatter_unpol.pdf',additional_artists=art_u,bbox_inches="tight")
fig_full.savefig(where_save + '/scatter_full.pdf',additional_artists=art_f,bbox_inches="tight")

print('figures saved into '+where_save)

###################scatter Diff
#######longitudinal
fig_Dlong = plt.figure(5)
plt.title('Longitudinal ')
plt.legend(loc='lower right', ncol=2, fancybox=True, fontsize='small')
plt.xlabel('truth cos'+r'$\theta$')
plt.ylabel('reconstructed - truth cos'+r'$\theta$')

# #########transverse
fig_Dtrans = plt.figure(6)
plt.title('Transverse')
plt.legend(loc='lower right', ncol=2, fancybox=True, fontsize='small')
plt.xlabel('truth cos'+r'$\theta$')
plt.ylabel('reconstructed - truth cos'+r'$\theta$')

# #######unpolarized
fig_Dunpol = plt.figure(7)
plt.title('Unpolarized')
plt.legend(loc='lower right', ncol=2, fancybox=True, fontsize='small')
plt.xlabel('truth cos'+r'$\theta$')
plt.ylabel('reconstructed - truth cos'+r'$\theta$')

# ######full computation
fig_Dfull = plt.figure(8)
plt.title('Full computation')
plt.legend(loc='lower right', ncol=2, fancybox=True, fontsize='small')
plt.xlabel('truth cos'+r'$\theta$')
plt.ylabel('reconstructed - truth cos'+r'$\theta$')

fig_Dlong.savefig(where_save + '/Diffscatter_long.pdf')
fig_Dtrans.savefig(where_save + '/Diffscatter_trans.pdf')
fig_Dunpol.savefig(where_save + '/Diffscatter_unpol.pdf')
fig_Dfull.savefig(where_save + '/Diffscatter_full.pdf')

print('figures saved into '+where_save)