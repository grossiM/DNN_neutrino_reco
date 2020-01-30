import os
import sys
import configparser
import warnings

import pandas as pd
import numpy as np
import fnmatch
import pickle

from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt

repo = os.environ['RECO_REPO']
sys.path.append(repo + '/neutrinoreconstruction/DeepLearning/DataHandler')

import optimizeThr as ot
import handler_kinematics as kinematics
import tables

class GridEvaluation():

    def __init__(self, config_file, selection=''):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.config.read(config_file)
        
        self.trained_models = self.config.get('output','output-folder')
        training_variables = self.config.get('training', 'training-variables').split(',')
        discriminating_variable = self.config.get('training', 'discriminating-variable')

        print(">>> Loading datasets ... ")

        pd_val_frames = []
        for val_sample in self.config.get('input', 'data-val').split(','):
            pd_val_frames.append(pd.read_hdf(val_sample))
        self.pd_valid = pd.concat(pd_val_frames)
        self.data_valid = self.pd_valid[training_variables].values
        self.truth_valid = self.pd_valid[discriminating_variable].values

        self.pd_data_long = pd.read_hdf(self.config.get('evaluation','long-data'))
        self.pd_data_trans = pd.read_hdf(self.config.get('evaluation','trans-data'))
        self.data_long = self.pd_data_long[training_variables].values
        self.data_trans = self.pd_data_trans[training_variables].values
        self.truth_long = self.pd_data_long['truth_cos_theta'].values
        self.truth_trans = self.pd_data_trans['truth_cos_theta'].values
        self.truth_unpol = self.pd_valid['truth_cos_theta'].values

        if self.config.get('evaluation','type')=='binary': 
            self.fig_roc = plt.figure(1)

        dirs = os.listdir(self.config.get('output','output-folder'))
        self.dirs = fnmatch.filter(dirs, '*'+selection+'*')
        print(">>> Input models: " + "\n" + "\n".join(self.dirs) + "\n")

        self.evaluate_all()

        if self.config.get('evaluation','type') == 'binary':
            plt.figure(1)
            plt.legend(loc='upper right', ncol=2, fancybox=True, fontsize='small')
            plt.ylabel('fakes')
            plt.ylabel('efficiency')
            plt.title('ROC curves')
            self.fig_roc.savefig(self.config.get('evaluation', 'output') + '/roc_curves.pdf')

        self.pd_data_long.to_hdf(self.config.get('evaluation', 'output')+'/calibrated_long.h5', 'calibrated_long', mode='w', table=True)
        self.pd_data_trans.to_hdf(self.config.get('evaluation', 'output')+'/calibrated_trans.h5', 'calibrated_trans', mode='w', table=True)              
        self.pd_valid.to_hdf(self.config.get('evaluation', 'output')+'/calibrated_valid.h5', 'calibrated_valid', mode='w', table=True)

    #########################################################################

    def roundScore(self, score, thr):
        indeces = np.argwhere(score > thr)
        for index in indeces:
            score[index] = 1
        indeces = np.argwhere(score <= thr)
        for index in indeces:
            score[index] = 0
        score = score.astype(int)

        return score

    #####################################################################

    def evaluate_all(self):
        for model_dir in self.dirs:
            models = os.listdir(self.config.get('output','output-folder') + '/' + model_dir)
            models = fnmatch.filter(models, self.config.get('evaluation','model-of-interest'))
            if len(models) == 0:
                raise ValueError('No models mathcing pattern '+self.config.get('evaluation','model-of-interest')+' found in '+model_dir)
            if len(models) > 1 and self.config.get('evaluation', 'type') == 'binary':
                warnings.warn('Only '+models[-1]+' score will be rounded')
            for model_ep in models:
                self.evaluate( model_dir, model_ep)
    
    def evaluate(self, model_dir, model_ep):
        path = self.trained_models + '/' + model_dir
        
        model_name = path + '/' + model_ep
        model = load_model(model_name)

        scaler_name = path + '/scaler.pkl'
        scaler = joblib.load(scaler_name)
        data_scaled_long = scaler.transform(self.data_long)
        data_scaled_trans = scaler.transform(self.data_trans)
        data_scaled_valid = scaler.transform(self.data_valid)

        pred_long = model.predict(data_scaled_long)
        pred_trans = model.predict(data_scaled_trans)
        pred_valid = model.predict(data_scaled_valid)

        if int(self.config.get('output','save-steps'))==1:
            epoch = model_ep[19:]
            print(">>> Evaluating model " + model_dir + " (epoch " + epoch + ") ... ")
            self.pd_data_long[model_dir+'_e'+epoch] = pred_long
            self.pd_data_trans[model_dir+'_e'+epoch] = pred_trans
            self.pd_valid[model_dir+'_e'+epoch] = pred_valid
            model_label = model_dir + '_e' + epoch
        else: 
            print(">>> Evaluating model " + model_dir + " ... ")
            self.pd_data_long[model_dir+'_pred'] = pred_long
            self.pd_data_trans[model_dir+'_pred'] = pred_trans
            self.pd_valid[model_dir+'_pred'] = pred_valid
            model_label = model_dir

        if self.config.get('evaluation', 'type') == 'binary':
            plt.figure(1)
            auc = roc_auc_score(self.truth_valid, pred_valid)
            print(">>> AUC: ",auc)
            fp , tp, th = roc_curve(self.truth_valid, pred_valid)
            thr = ot.optimizeThr(fp,tp,th)
            plt.plot(fp, tp, label=model_label)

            self.pd_data_long[model_dir+'_rounded_score'] = self.roundScore(pred_long, thr)
            self.pd_data_trans[model_dir+'_rounded_score'] = self.roundScore(pred_trans, thr)
            selection_valid = self.roundScore(pred_valid, thr)
            self.pd_valid[model_dir+'_rounded_score'] = selection_valid

            nall = selection_valid.shape[0]
            comparison = np.ones((nall,1), dtype=bool)
            np.equal(np.expand_dims(self.truth_valid, 1),selection_valid,comparison)
            print(">>> Fraction of correct predictions: "+str(np.sum(comparison)/nall))
