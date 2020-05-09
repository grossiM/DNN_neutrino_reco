#!/usr/bin/env python3
#################################
# M. Grossi - J.Novak #2019
################################
#macro that search for root hist in folder path and create npz files with specific name convention
########################
import numpy as np
import argparse
import math
import fnmatch
import os

from ROOT import TFile, TH1F
from root_numpy import hist2array

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path_folder', type=str, required=False, default='/Users/it058990/Downloads')

args = parser.parse_args()

#here implement looking for root hist
base_name = args.path_folder
if not os.path.exists(base_name):
    raise ValueError('Error: folder '+base_name+' does not exists')

hist_list = []
form_list = ['TH1','TH2','TH3','THnBase']

for i in (fnmatch.filter(os.listdir(base_name), '*root')):
    hist_list.append(i)
#loop over all root hist and transform to npz
print(hist_list)

for l in hist_list:
    print(l)
    a = TFile.Open(base_name+'/'+l)
    title = l.replace('.root','')
#estraggo nome histogramma, esempio h1, converto a numpy e salvo
    for m in range(len(a.GetListOfKeys())):
    
        for n in form_list:
            if (a.Get(a.GetListOfKeys()[m].GetName()).InheritsFrom(n)):
                h1 = a.Get(a.GetListOfKeys()[m].GetName())
                h1p = hist2array(h1)
                np.savez(base_name + '/h_' + title + '_' + a.GetListOfKeys()[m].GetName(), h1p)
                print('Saving histo ' + title + '_' +  str(a.GetListOfKeys()[m].GetName()) +  ' '+  'in: ' + base_name)
            else:
                #print('skip this conversion: ' + a.GetListOfKeys()[m].GetName())
                None
    del h1,h1p
 
#http://scikit-hep.org/root_numpy/reference/generated/root_numpy.hist2array.html
#https://codereview.stackexchange.com/questions/158004/reading-and-processing-histograms-stored-in-root-files