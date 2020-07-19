#!/usr/bin/env python3
#################################
# M. Grossi - J.Novak # May 2020
################################
#macro that search for root hist in folder path and create npz files and pdf plot with specific name convention
########################
'''
python3 hist_root_python_converter.py -c plt_hist.cfg -s 0

python3 hist_root_python_converter.py -c plt_hist.cfg -s 1
--> example insert: 7, 8, 9, 10, 11, 12

'''
import numpy as np
import argparse
import math
import fnmatch
import configparser

import os
import uproot 
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
parser.add_argument('-s', '--step', type=int, required=True)

args = parser.parse_args()
step = args.step

config = configparser.ConfigParser()
config.optionxform = str
config.read(args.config)

where_save =  config.get('output','output-folder')
base_name = config.get('input','data-root')
save_npz = int(config.get('plotting','save-npz'))
save_plot = int(config.get('plotting','save-pdf-plot'))


if not os.path.exists(base_name):
    raise ValueError('Error: folder '+base_name+' does not exists')

hist_list = []
form_list = ["<class 'uproot.rootio.TH1F'>"]
#, "<class 'uproot.rootio.TH2F'>","<class 'uproot.rootio.TH3F'>"]
 
for i in (fnmatch.filter(os.listdir(base_name), '*root')):
    hist_list.append(i)
#loop over all root hist and transform to npz
print(hist_list)

for l in hist_list:
    
    a = uproot.open(base_name+'/'+l)
    #a = TFile.Open(base_name+'/'+l)
    title = l.replace('.root','')

#check class TH1F and then save histo

#####step 0 print all elements (histogram type) inside the root file so that you can choose a list of what to print, save, plot
    if step == 0:
        print(l)
        sel_list = []
        nsel_list = []
        for m, item in enumerate(a.keys(), start=0):
        #for m in range(len(a.keys())):
            for n in form_list:
                #implement a better check on the class like: isinstance(b,type(a)) where b is the file[hist] and type(a) is class 'uproot.rootio.TH1F', now we compare a string
                if (str(type(a[a.keys()[m]]))== n):
                    sel_list.append(a.keys()[m] )
                    nsel_list.append(m)
                    print(m, a.keys()[m])
                else:
                    None

        print('Please select a list of plot according to the convention above')
        print(nsel_list)

####in step 1 you plot inserting the list
    if step == 1: 
        print(config.get('plotting','indices'))
        if config.get('plotting','indices') == '':
            value = input("Please enter indices of hist to plot:\n").split(',')
        else:
            value = config.get('plotting','indices').split(',')
        for index, m in enumerate(value):
                #implement a better check on the class like: isinstance(b,type(a)) where b is the file[hist] and type(a) is class 'uproot.rootio.TH1F', now we compare a string
                h1 = a[a.keys()[int(m)]]

                lab = a.keys()[int(m)].decode('ASCII')
                n_sel = [int(s) for s in lab if s.isdigit()] 
                label = n_sel[0]
                if len(n_sel)==1:
                    n_sel = 'random'
                else:
                    n_sel = n_sel[0]

                entry = 'selection '+ str(n_sel)
                new_label = lab.replace(';1','')

                bins = h1.edges
                counts = h1.values
                assert len(bins) == len(counts) + 1
                centroids = (bins[1:] + bins[:-1]) / 2
                #counts_, bins_, _ = plt.hist(centroids, bins=len(counts),weights=counts, range=(min(bins), max(bins)))
                #assert np.allclose(bins_, bins)
                #assert np.allclose(counts_, counts)
                if config.get('plotting','entries') == '':
                    entry = input("Please enter legend entry for hist "+m+":")
                else:
                    entry = config.get('plotting','entries').split(',')[index]
                
                plt.figure(1)
                plt.legend()
                first_color = config.get('plotting','first-color')
                if index == 0:
                    color = 'C' + first_color
                    h1p = plt.hist(centroids, bins=len(counts), weights=counts, range=(min(bins), max(bins)), histtype='stepfilled', facecolor='w', hatch='//', edgecolor=color, density=False, linewidth=2, label=entry)
                else:
                    color = 'C' + str(index)
                    if (index < int(first_color) + 1):
                        color = 'C' + str(index-1)
                    h1p = plt.hist(centroids, bins=len(counts),weights=counts, range=(min(bins), max(bins)),label=entry, density=False, histtype='step', linewidth=2, edgecolor=color)
                if save_npz:
                    np.savez(base_name + '/hNEW_' + title + '_' + new_label, h1p= h1p)
                    print('Saving histo ' + title + '_' +  new_label +  ' '+  'in: ' + base_name)
                   
        if save_plot:
            plt.figure(1)
            art = []
            lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15),ncol= int(config.get('legend','ncol')), fancybox=True, fontsize= int(config.get('legend','fontsize')))
            art.append(lgd)
            if config.get('plotting','label') != '':
                raw_string = eval('"' + config.get('plotting','label') + '"')
                xmin, xmax = plt.xlim()
                ymin, ymax = plt.ylim()
                plt.ylim((ymin,1.1*ymax))
                plt.annotate(raw_string,xy=(xmin+0.1*(xmax-xmin), ymax),fontsize=14,weight='bold')
            plt.xlabel(config.get('plotting','xlabel'))
            plt.ylabel(config.get('plotting','ylabel'))
            plt.savefig(base_name + '/' + config.get('plotting','name-pdf')+ '.pdf', additional_artists=art,bbox_inches="tight")
            plt.savefig(base_name + '/' + config.get('plotting','name-pdf')+ '.eps', additional_artists=art,bbox_inches="tight", format='eps')
            print('Saving plot ' + base_name + '/' + config.get('plotting','name-pdf')+ '.pdf')