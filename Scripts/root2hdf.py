import uproot
import pandas as pd
import numpy as np
import argparse
import sys

# upfile = uproot.open('/ceph/grid/home/atlas/jakobn/bbWW-Analysis/boosted1lep/ForJakob/HHbbWW_X2000_1lep.root')
# upfile = uproot.open('/afs/f9.ijs.si/home/jakobn/CxAODFramework_HHbbVV/run/signal_DNNtree/data-MVATree/NonRes_HHbbVV.root')
# upfile = uproot.open('/afs/f9.ijs.si/home/jakobn/CxAODFramework_HHbbVV/run/ttbar_DNNtree/data-MVATree/ttbar_nonallhad.root')

parser = argparse.ArgumentParser()
parser.add_argument('-in', '--input', type=str, required=False)
parser.add_argument('-o', '--output-folder', type=str, required=True)#output folder 
parser.add_argument('-n', '--name', type=str, required=True)#basename of the file
parser.add_argument('-s', '--separation', type=str, required=False, default='1')# ex. '0.2:0.4:0.4' separation of dataset between training, test, evaluation 
args = parser.parse_args()

upfile = uproot.open(args.input)
uptree = upfile['Nominal']
pdarray = uptree.pandas.df()
pdrandom = pdarray.sample(frac=1)
pdrandom = pdrandom.query('ValidTruthEvent == 1')

n_events = pdrandom.shape[0]

fractions = [float(i) for i in args.separation.split(':')]
if np.sum(fractions) != 1:
    print('ERROR: split fraction must sum to 1, like 0.2:0.4:0.4 ')
    sys.exit(1)

fractions = [round(x * n_events) for x in fractions]
event_difference = n_events - np.sum(fractions)
fractions[0] = fractions[0] + event_difference #correction to have right number of events

curr_ind = 0
indeces = [curr_ind]
for fraction in fractions:
    curr_ind += fraction
    indeces.append(curr_ind) 

for i in range(len(indeces)-1):
    data = pdrandom[indeces[i]:indeces[i+1]]
    name = args.output_folder + '/' + args.name
    if i == 0:
        name = name +'_train'
    elif i == 1:
        name = name +'_val'
    elif i == 2:
        name = name + '_eval'
    #add hf5
    #print(data.columns)

    data.to_hdf(name+'.h5',name,mode='w',table=True)#to check
'''
pdtrain = pdrandom[:2300]
pdval = pdrandom[2300:4600]
pdeval = pdrandom[4600:]
pdrandom.to_hdf('signal_r3.h5','signal',mode='w',table=True)
pdtrain.to_hdf('signal_train_r3.h5','signal',mode='w',table=True)
pdval.to_hdf('signal_val_r3.h5','signal',mode='w',table=True)
pdeval.to_hdf('signal_eval_r3.h5','signal',mode='w',table=True)
'''
