#!/usr/bin/env python3
"""

  Michele Grossi <michele.grossi@it.ibm.com>
  Jakob Novak <janob.novak.cern.ch>
  Version 1.0, June 2020
  Create a text file with output result

  USAGE: python3  xsquare_calc.py -c JobOption/NNplot_config.cfg
  """

import os
import sys
import configparser

import fnmatch
import re
import pandas as pd
import numpy as np
import argparse
from scipy.stats import chisquare
from sklearn.metrics import mean_squared_error


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
args = parser.parse_args()

config = configparser.ConfigParser()
config.optionxform = str
config.read(args.config)

where_save = config.get('output','output-folder')

outfile= open(where_save+"/xsquare.txt","w+")
#this part should be implemented if the for cicle to change the folder name according to all selection list

truth = config.get('input','truth-label')
###data reading & checking
####hdf5 reading
for f in config.get('input','data').split(','):  
    print(f)
    hdf_f = pd.read_hdf(f)
    #############looping through all model
    outfile.write('File: ')
    outfile.write(f)
    outfile.write("\n")
    print('looping through all models:')
    for i in fnmatch.filter(hdf_f.columns, '*' + config.get('input','model_sel') + '*_e100'):
        chi_statistic, p_value = chisquare(hdf_f[i], hdf_f[truth])
        rmse = mean_squared_error(hdf_f[truth],hdf_f[i], squared=False)
        outfile.write("model: ")
        outfile.write(str(i))
        outfile.write("\n")
        outfile.write("Xsquare = {:.3f} \n".format(round(chi_statistic, 3)))
        #outfile.write("\n")
        outfile.write("rmse = {:.3f}\n".format(round(rmse, 3)))
        #outfile.write("\n")
print('file saved in: ' + where_save )
      
    
