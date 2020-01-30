import os
import sys
import configparser

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
sys.path.append(repo + 'neutrinoreconstruction/DeepLearning/DataHandler')

import optimizeThr as ot
import handler_kinematics as kinematics




class GridEvaluation():

    def __init__(self, config_file, selection=''):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.config.read(config_file)
        
        #to uncomment if we manage the creation of directory (calling one time the class, now it's implemented in Evaluate.py)
        #folder_name = self.config.get('evaluation','output')
        #if os.path.exists(folder_name):
        #    raise ValueError('Error: folder '+folder_name+' already exists')
        #os.system('mkdir ' + folder_name)
        
        self.trained_models = self.config.get('output','output-folder')
        training_variables = self.config.get('training', 'training-variables').split(',')
        discriminating_variable = self.config.get('training', 'discriminating-variable')

        pd_val_frames = []
        for val_sample in self.config.get('input', 'data-val').split(','):
            pd_val_frames.append(pd.read_hdf(val_sample))
        pd_valid = pd.concat(pd_val_frames)
        self.data_valid = pd_valid[training_variables].values
        self.truth_valid = pd_valid[discriminating_variable].values

        pd_data_long = pd.read_hdf(self.config.get('evaluation','long-data'))
        pd_data_trans = pd.read_hdf(self.config.get('evaluation','trans-data'))
        self.data_long = pd_data_long[training_variables].values
        self.data_trans = pd_data_trans[training_variables].values
        self.truth_long = pd_data_long['truth_cos_theta'].values
        self.truth_trans = pd_data_trans['truth_cos_theta'].values
        self.truth_unpol = pd_valid['truth_cos_theta'].values

        if self.config.get('evaluation','type')=='binary': 
            self.angle_unpol = self.getAngle(pd_valid)
            self.angle_long = self.getAngle(pd_data_long)
            self.angle_trans = self.getAngle(pd_data_trans)
            self.fig_roc = plt.figure(1)

        dirs = os.listdir(self.config.get('output','output-folder'))
        self.dirs = fnmatch.filter(dirs, '*'+selection+'*')
        print("Input models: " + "\n" + "\n".join(self.dirs) + "\n")

        binning = self.config.get('evaluation', 'binning')
        binning = binning.replace('\\', '')
        self.bins = binning.split(',')

        self.fig_long = plt.figure(2)
        h_long_true = plt.hist(self.truth_long, np.arange(float(self.bins[0]), float(self.bins[1]), float(self.bins[2])), alpha = 0.5, edgecolor='black', linewidth=2.1, label='truth')

        self.fig_trans = plt.figure(3)
        h_trans_true = plt.hist(self.truth_trans, np.arange(float(self.bins[0]), float(self.bins[1]), float(self.bins[2])), alpha = 0.5, edgecolor='black', linewidth=2.1, label='truth')

        self.fig_unpol = plt.figure(9)
        h_unpol_true = plt.hist(self.truth_unpol, np.arange(float(self.bins[0]), float(self.bins[1]), float(self.bins[2])), alpha = 0.5, edgecolor='black', linewidth=2.1, label='truth')

        np.savez(self.config.get('evaluation', 'output') + '/h_truth',
                 unpol=h_unpol_true, trans=h_trans_true, long=h_long_true)

        self.evaluate_all()

        plt.figure(2)
        plt.legend(loc='upper left', ncol=3, fancybox=True, fontsize='small')
        plt.xlabel('cos'+r'$\theta$')
        plt.ylabel('Number of events')
        plt.title('Longitudinal polarization')
        self.fig_long.savefig(self.config.get('evaluation', 'output') + '/theta_long.pdf')

        plt.figure(3)
        plt.legend(loc='upper right', ncol=2, fancybox=True, fontsize='small')
        plt.xlabel('cos'+r'$\theta$')
        plt.ylabel('Number of events')
        plt.title('Transverse polarization')
        self.fig_trans.savefig(self.config.get('evaluation', 'output') + '/theta_trans.pdf')

        plt.figure(9)
        plt.legend(loc='upper right', ncol=2, fancybox=True, fontsize='small')
        plt.xlabel('cos'+r'$\theta$')
        plt.ylabel('Number of events')
        plt.title('Unpolarized')
        self.fig_unpol.savefig(self.config.get('evaluation', 'output') + '/theta_unpol.pdf')


        # #####
        self.fig_trans_diff = plt.figure(4)

        plt.figure(4)
        plt.legend(loc='upper right', ncol=2, fancybox=True, fontsize='small')
        
        plt.ylabel('Difference in Number of events')
        plt.title('Transverse polarization difference truth - reco')
        self.fig_trans_diff.savefig(self.config.get('evaluation', 'output') + '/diff_theta_trans.pdf')

        # #####
        self.fig_long_diff = plt.figure(5)

        plt.figure(5)
        plt.legend(loc='upper right', ncol=2, fancybox=True, fontsize='small')
        
        plt.ylabel('Difference in Number of events')
        plt.title('Longitudinal polarization difference truth - reco')
        self.fig_long_diff.savefig(self.config.get('evaluation', 'output') + '/diff_theta_long.pdf')


        self.fig_long_diff_cos = plt.figure(6)

        plt.figure(6)
        plt.legend(loc='upper right', ncol=2, fancybox=True, fontsize='small')
        plt.xlabel('cos'+r'$\theta$')
        plt.ylabel('cos'+r'$\theta$' + 'reconstructed')
        plt.title('Longitudinal polarization reconstruction vs truth')
        self.fig_long_diff_cos.savefig(self.config.get('evaluation', 'output') + '/diff_theta_long_cos.pdf')


        self.fig_trans_diff_cos = plt.figure(7)

        plt.figure(7)
        plt.legend(loc='upper right', ncol=2, fancybox=True, fontsize='small')
        plt.xlabel('cos'+r'$\theta$')
        plt.ylabel('cos'+r'$\theta$' + 'reconstructed')
        plt.title('Transverse polarization polarization reconstruction vs truth')
        self.fig_trans_diff_cos.savefig(self.config.get('evaluation', 'output') + '/diff_theta_trans_cos.pdf')

        ####################################
        if self.config.get('evaluation','type') == 'binary':
            plt.figure(1)
            plt.legend(loc='upper right', ncol=2, fancybox=True, fontsize='small')
            plt.ylabel('fakes')
            plt.ylabel('efficiency')
            plt.title('ROC curves')
            self.fig_roc.savefig(self.config.get('evaluation', 'output') + '/roc_curves.pdf')

    def getAngle(self, pdarray):
        variables0 = ['mu_px', 'mu_py', 'mu_pz', 'mu_E', 'v_mu_px', 'v_mu_py', 'v_mu_sol0', 'v_mu_predict_E0']
        variables1 = ['mu_px', 'mu_py', 'mu_pz', 'mu_E', 'v_mu_px', 'v_mu_py', 'v_mu_sol1', 'v_mu_predict_E1']
        cos_theta0 = kinematics.cos_theta(pdarray, variables0)
        cos_theta1 = kinematics.cos_theta(pdarray, variables1)
        cos_theta0 = np.expand_dims(cos_theta0, 1)
        cos_theta1 = np.expand_dims(cos_theta1, 1)
        print("my cos theta shape: ", cos_theta0.shape)
        return np.concatenate((cos_theta0,cos_theta1),1)

    def scoreToVar(self, score, solutions, thr):
        indeces = np.argwhere(score > thr)
        for index in indeces:
            score[index] = 1
        indeces = np.argwhere(score <= thr)
        for index in indeces:
            score[index] = 0
        score = score.astype(int)

        return [solutions[i, sign[0]] for i, sign in enumerate(score)]
    #####################################################################
    def evaluate_all(self):
        self.diff_dict_long = {}#create a dictionary to be filled with difference
        self.diff_dict_trans = {}
        for model_dir in self.dirs:
            self.evaluate(model_dir)
        self.diff_dict_long['truth'] = self.truth_long[0::1000]
        self.diff_dict_trans['truth'] = self.truth_trans[0::1000]

        print(self.diff_dict_trans)
        self.plot_diff_dic(self.diff_dict_long,6, 'long')
        self.plot_diff_dic(self.diff_dict_trans,7, 'trans')



    def plot_diff_dic(self, dict,fig, pol_type = 'long'):
        plt.figure(fig)
        for key in dict:
            x = dict['truth']
            y = dict[key] 
            sc = plt.scatter(x,y,label=key)
            plt.legend()
            if pol_type == 'long':
                np.savez(self.config.get('evaluation', 'output') + '/h_' + key, scatter_long = sc)
            elif pol_type == 'trans':
                np.savez(self.config.get('evaluation', 'output') + '/h_' + key, scatter_trans = sc)
            else:
                print('wrong polarizarion')

    
    def evaluate(self, model_dir):
        path = self.trained_models + '/' + model_dir
        
        if(int(self.config.get('output','save-steps'))==1):
            #print('save-steps=1')
            model_name = path + '/current_model_epoch' + str(self.config.get('training','epochs'))
        else: 
            model_name = path + '/current_model'        
            print('not save-steps=',self.config.get('output','save-steps'))
        model = load_model(model_name)

        scaler_name = path + '/scaler.pkl'
        scaler = joblib.load(scaler_name)
        data_scaled_long = scaler.transform(self.data_long)
        data_scaled_trans = scaler.transform(self.data_trans)
        data_scaled_valid = scaler.transform(self.data_valid)

        pred_long = model.predict(data_scaled_long)
        pred_trans = model.predict(data_scaled_trans)
        pred_valid = model.predict(data_scaled_valid)
        ###############################
        #evaluating the difference truth - reco
        diff_long = self.truth_long - pred_long[:,0]
        predlong = pred_long[:,0]
        diff_long = diff_long[0::1000]
        diff_trans = self.truth_trans - pred_trans[:,0]
        predtrans = pred_trans[:,0]
        diff_trans = diff_trans[0::1000]
        self.diff_dict_long[model_dir] = predlong[0::1000]
        self.diff_dict_trans[model_dir] = predtrans[0::1000]

        # self.diff_dict_long[model_dir] = diff_long
        # self.diff_dict_trans[model_dir] = diff_trans
        
        ##############################
        #test_stat_long = ks_2samp(self.truth_long, pred_long[:,0])
        #print('Kolmogorov-Smirnov 2-way test longitudinal component: {}'.format(test_stat_long))
        #test_stat_trans = ks_2samp(self.truth_trans, pred_trans[:,0])
        #print('Kolmogorov-Smirnov 2-way test transverse component for model{} is: {}'.format(model_name, test_stat_trans))
        
        #chisq_l, p_l = chisquare(self.truth_long, f_exp=pred_long[:,0])
        #print('Chi-square test: chi-square value for longitudinal component: {}, p-value: {}'.format(chisq_l, p_l))
        #chisq_t, p_t = chisquare(self.truth_trans, f_exp=pred_trans[:,0])
        #print('Chi-square test: chi-square value for transverse component: {}, p-value: {}'.format(chisq_t, p_t))
        ##############################

        #self.kolmog_smirnov(self.truth_long, pred_long[:,0])
        #self.kolmog_smirnov(self.truth_trans, pred_trans[:,0])
        #self.chisqr(self.truth_long, pred_long[:,0])
        #self.chisqr(self.truth_trans, pred_trans[:,0])

        if self.config.get('evaluation', 'type') == 'binary':
            plt.figure(1)
            auc = roc_auc_score(self.truth_valid, pred_valid)
            print(">>> AUC: ",auc)
            fp , tp, th = roc_curve(self.truth_valid, pred_valid)
            plt.plot(fp, tp, label=model_dir)

            thr = ot.optimizeThr(fp,tp,th)

            plt.figure(2)
            h_long = plt.hist(self.scoreToVar(pred_long, self.angle_long, thr), np.arange(float(
                self.bins[0]), float(self.bins[1]), float(self.bins[2])), alpha=0.3, label=model_dir)

            plt.figure(3)
            h_trans = plt.hist(self.scoreToVar(pred_trans, self.angle_trans, thr), np.arange(float(
                self.bins[0]), float(self.bins[1]), float(self.bins[2])), alpha=0.3, label=model_dir)

            plt.figure(9)
            h_unpol = plt.hist(self.scoreToVar(pred_valid, self.angle_unpol, thr), np.arange(float(
                self.bins[0]), float(self.bins[1]), float(self.bins[2])), alpha=0.3, label=model_dir)

            np.savez(self.config.get('evaluation', 'output') + '/h_' + model_dir, unpol=h_unpol, trans=h_trans, long=h_long)

        if self.config.get('evaluation','type')=='regression':
            
            plt.figure(2)
            h_long = plt.hist(pred_long, np.arange(float(self.bins[0]), float(self.bins[1]), float(self.bins[2])), alpha = 0.3, label=model_dir)

            plt.figure(3)
            h_trans = plt.hist(pred_trans, np.arange(float(self.bins[0]), float(self.bins[1]), float(self.bins[2])), alpha = 0.3, label=model_dir)

            plt.figure(9)
            h_unpol = plt.hist(pred_valid, np.arange(float(self.bins[0]), float(self.bins[1]), float(self.bins[2])), alpha = 0.3, label=model_dir)


            plt.figure(4)
            plt.grid(True)
            d_trans = plt.hist(diff_trans, np.arange(float(self.bins[0]), float(self.bins[1]), float(self.bins[2])), alpha = 0.5, label=model_dir)

            plt.figure(5)
            plt.grid(True)
            d_long = plt.hist(diff_long, np.arange(float(self.bins[0]), float(self.bins[1]), float(self.bins[2])), alpha = 0.5, label=model_dir)

            np.savez(self.config.get('evaluation', 'output') + '/h_' + model_dir, unpol=h_unpol, trans=h_trans, long=h_long, dlong = d_long, dtrans = d_trans)