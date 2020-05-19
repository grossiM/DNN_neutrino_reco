#################################
# M. Grossi - J.Novak #2019
################################
#macro to create 3 preprocessed data (h5 format) from root file
#all branches included that can be then selected in training phase
#input root file - output 3 h5 file + 1 pickle dataset (data_handler)
#USAGE: python3 create_preprocessData.py -in '/Users/it058990/Desktop/MicheleG/PhD/VBS_COST_Group/GITBicocca/DNN_storage/data/rootfiles/gen81_mu_ewk_semilept_lsf_lhe.root'
#  -o '/Users/it058990/Desktop/MicheleG/PhD/VBS_COST_Group/GITBicocca/DNN_storage/data/rootfiles' -n 'gen81_mu_ewk_semilept_lsf_lhe' -ch 0 -s ‘0.3:0.3:0.4’
# TRAINING: unpolarized, transverse, longitudinal, mixed(transverse+longitudinal)
#evaluation: polarized(transverse and longitudinal), unpolarized (to then perform fit and deduce fraction of polarization)

import sys
import numpy as np
import os
import pickle
import copy
import argparse
import tables

repo= os.environ['NEW_REPO']
sys.path.append(repo + '/DNN_neutrino_reco/Utils/DataHandler')
import DataHandler

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

#this part can be improved avoiding too many line command
parser = argparse.ArgumentParser()
parser.add_argument('-in', '--input', type=str, required=False)
parser.add_argument('-o', '--output-folder', type=str, required=True)#output folder 
parser.add_argument('-n', '--name', type=str, required=True)#basename of the file
parser.add_argument('-ch', '--channel', type=int, required=True)#channel type if semilept =0 full lep =1
parser.add_argument('-s', '--separation', type=str, required=False, default='1')# ex. '0.2:0.4:0.4' separation of dataset between training, test, evaluation 
parser.add_argument('-nev', '--nevents', type=int, required=False, default=-1)
args = parser.parse_args()

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

print(">>> Loading datasets...")

event_variables = []
event_variables.append(['mu_px', 'mu_py', 'mu_pz', 'mu_E'])#[particles variables]
event_variables.append(['q_fin_px1', 'q_fin_py1', 'q_fin_pz1', 'q_fin_E1'])
event_variables.append(['q_fin_px2', 'q_fin_py2', 'q_fin_pz2', 'q_fin_E2'])

if args.channel == 1:
    
    print('full leptonic channel sample')
    #we keep for real full leptonic analysis
    columns={}
    event_variables.append(['el_px', 'el_py', 'el_pz', 'el_E']) 
    event_variables.append(['v_el_px', 'v_el_py', 'v_el_pz', 'v_el_E'])

else:
    print('semi leptonic channel sample')
    columns={}

    event_variables.append(['q_fin_px3', 'q_fin_py3', 'q_fin_pz3', 'q_fin_E3'])#semileptonic
    event_variables.append(['q_fin_px4', 'q_fin_py4', 'q_fin_pz4', 'q_fin_E4'])##semileptonic

##########################################################################
data_handler = DataHandler.DataHandler(args.input, 'tree', True, columns)
##########################################################################
# #muon part
data_handler.calcCosTheta(['mu_px', 'mu_py', 'mu_pz', 'mu_E', 'v_mu_px', 'v_mu_py', 'v_mu_pz', 'v_mu_E'],'mu_truth')
#data_handler.appendMass(variables_sol1m,'event1_mu')#in a different way
data_handler.getPtEtaPhi('mu')
data_handler.getPtEtaPhi('v_mu')
# #electron part
data_handler.calcCosTheta(['el_px', 'el_py', 'el_pz', 'el_E', 'v_el_px', 'v_el_py', 'v_el_pz', 'v_el_E'],'el_truth')
data_handler.getPtEtaPhi('el')
data_handler.getPtEtaPhi('v_el')
# #####sum of both neutrino transverse momenta is measurable
data_handler.getPtvv(['v_mu_pt','v_el_pt'])

#dropping branch according to selected channel
#BE CAREFUL!!! LEPTONIC BRANCHES WILL BE RENAMED TO MIMIC SEMILEPTONIC CHANNEL (THIS PART MUST BE DELETED WHEN STUDING REAL FULL LEPTONIC CASE)
if args.channel == 1:
    data_handler.pdarray = data_handler.pdarray.drop(labels=['mu_p4','v_mu_p4','el_p4','v_el_p4','q_init1_p4','q_init2_p4','q_fin1_p4','q_fin2_p4'],axis=1)
else:
    data_handler.pdarray = data_handler.pdarray.drop(labels=['mu_p4','v_mu_p4','q_init1_p4','q_init2_p4','q_fin1_p4','q_fin2_p4','q_fin3_p4','q_fin4_p4'],axis=1)

#########
n_events = args.nevents

if n_events > data_handler.pdarray.shape[0]:
    raise ValueError('wrong number of events selected, the maximum number available for training is:' + str(data_handler.pdarray.shape[0]))
elif n_events == -1 :
    n_events = data_handler.pdarray.shape[0]


#create training and evaluation dataset from original file
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
    data = data_handler.pdarray[indeces[i]:indeces[i+1]]
    #name = args.output_folder + '/' + args.name + str(fractions[i]) +'_' +str(i)
    name = args.output_folder + '/' + args.name + str(fractions[i])
    if i == 0:
        name = name +'_train'
    elif i == 1:
        name = name +'_val'
    elif i == 2:
        name = name + '_eval'
    #add hf5
    print(data.columns)

    data.to_hdf(name+'.h5',name,mode='w',format ='table')
    h5file = tables.open_file(name+'.h5',driver="H5FD_CORE")#this save data on disk after closure of python
    h5file.close()
     #in training I will need to open, load and work on data but not save it on disk:
    #h5file = tables.open_file("sample.h5", "a", driver="H5FD_CORE",driver_core_backing_store=0)
    
   
    print(name)

