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
#from sklearn.externals import joblib
from joblib import dump, load
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt

repo = os.environ['NEW_REPO']
sys.path.append(repo + '/DNN_neutrino_reco/Utils/DataHandler')

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
        training_labels = self.config.get('training', 'training-labels').split(',')

        print(">>> Loading datasets ... ")

        self.pd_names = []
        self.pd_eval = {}
        self.data_eval = {}
        self.truth_eval = {}
        counter = 0

        for eval_samples in self.config.get('evaluation', 'data-eval').split(':'):
            pd_eval_frames = []
            for eval_sample in eval_samples.split(','):
                pd_eval_frames.append(pd.read_hdf(eval_sample))
            pd_eval = pd.concat(pd_eval_frames)
            if len(pd_eval_frames) > 1:
                if (counter == 0):
                    sample_name = 'merged.calibrated.h5'
                else:
                    sample_name = 'merged'+str(counter)+'.calibrated.h5'
                counter = counter + 1
            if len(pd_eval_frames) == 1:
                sample_orig = eval_samples.split('/')[-1]
                base_name = sample_orig.split('.')
                base_name.insert(1,'calibrated')
                sample_name = '.'.join(base_name)
            self.pd_names.append(sample_name)
            self.pd_eval[sample_name] = pd_eval
            self.data_eval[sample_name] = pd_eval[training_variables].values
            self.truth_eval[sample_name] = pd_eval[training_labels].values

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

        for sample in self.pd_names:
            output_file = self.config.get('evaluation', 'output')+'/'+sample
            print(">>> Writing output "+output_file+" ...")
            #self.pd_eval[sample].to_hdf(output_file, 'evaluated_data', mode='w', table=True)
            self.pd_eval[sample].to_hdf(output_file, 'evaluated_data', mode='w', format='table')

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
                for sample in self.pd_names:
                    self.evaluate( model_dir, model_ep, sample)
    
    def evaluate(self, model_dir, model_ep, sample):
        path = self.trained_models + '/' + model_dir
        
        model_name = path + '/' + model_ep
        model = load_model(model_name)

        scaler_name = path + '/scaler.pkl'
        #scaler = joblib.load(scaler_name)
        scaler = load(scaler_name)

        data_scaled = scaler.transform(self.data_eval[sample])

        pred = model.predict(data_scaled)
        

        label_sc_name = path + '/label_scaler.pkl'
        #correction to use correct label scaler
        #label_sc_name = path + '/../label_scaler.pkl'

        if self.config.get('training','scale-label')== '1':
            if os.path.exists(label_sc_name):
                #label_scaler = joblib.load(label_sc_name)
                label_scaler = load(label_sc_name)
                orig_pred = pred[:10]
                pred = label_scaler.inverse_transform(pred)
                if (orig_pred == pred[:10]).all():
                    print('Error in label_scaler for model {0}'.format(model_dir))
                    sys.exit()
                else: print('Label_scaler OK')
            else: 
                print('scaler not found, exiting')
                sys.exit()

        if int(self.config.get('output','save-steps'))==1: # check this!
            epoch = model_ep[19:]
            print(">>> Evaluating model " + model_dir + " (epoch " + epoch + ") on sample " + sample.split('.')[0] + " ... ")
            if pred.shape[1]==1:
                self.pd_eval[sample][model_dir+'_e'+epoch] = pred
            else:
                for cat in range(pred.shape[1]):
                    self.pd_eval[sample][model_dir+'_cat'+str(cat)+'_e'+epoch] = pred[:,cat]
            model_label = model_dir + '_e' + epoch
        else: 
            print(">>> Evaluating model " + model_dir + " on sample " + sample.split('.')[0] + " ... ")
            if pred.shape[1]==1:
                self.pd_eval[sample][model_dir+'_pred'] = pred
            else:
                for cat in range(pred.shape[1]):
                    self.pd_eval[sample][model_dir+'_cat'+str(cat)+'_pred'] = pred[:,cat]
            model_label = model_dir

        if self.config.get('evaluation', 'type') == 'binary' and pred.shape[1]==1:
            plt.figure(1)
            auc = roc_auc_score(self.truth_eval[sample], pred)
            print(">>> AUC: ",auc)
            fp , tp, th = roc_curve(self.truth_eval[sample], pred)
            thr = ot.optimizeThr(fp,tp,th)
            plt.plot(fp, tp, label=model_label)

            selection = self.roundScore(pred, thr)
            # print('selection_shape: {}'.format(selection.shape))
            # print('truth_eval[sample] shape: {}'.format(self.truth_eval[sample].shape))

            self.pd_eval[sample][model_dir+'_rounded_score'] = selection

            nall = selection.shape[0]
            comparison = np.ones((nall,1), dtype=bool)
            #check truth_eval[sample] dimension due to possible different python version
            if self.truth_eval[sample].ndim == 1:
                np.equal(np.expand_dims(self.truth_eval[sample],1),selection,comparison)
            elif self.truth_eval[sample].ndim == 2:
                np.equal(self.truth_eval[sample],selection,comparison)
            else:
                np.equal(self.truth_eval[sample],selection,comparison)
            print(">>> Fraction of correct predictions: "+str(np.sum(comparison)/nall))
