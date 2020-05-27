import sys
import os
import configparser
import numpy as np
import pandas as pd 
import numpy as np

from shutil import copyfile

from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from keras import metrics
from keras import losses
from keras import optimizers
import keras.backend as K

import Grid
import Configurables
from plotting_func import plot_history
import Model
#import new_model
np.set_printoptions(threshold=sys.maxsize)

class TrainingHandler():

    def __init__(self, config_file):
        print('Initializing TrainingHandler...')
        self.properties = {}
        
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.config.read(config_file)
        Configurables.validate(self.config)
        self.readProperties()
        self.properties['unfold'] = int(self.properties['unfold'])
        self.properties['save-steps'] = int(self.properties['save-steps'])
        
        pd_train_frames = []
        pd_val_frames = []
        for train_sample in self.properties['data-train'].split(','):
            pd_train_frames.append(pd.read_hdf(train_sample))
        for val_sample in self.properties['data-val'].split(','):
            pd_val_frames.append(pd.read_hdf(val_sample))
        pd_train = pd.concat(pd_train_frames)
        pd_val = pd.concat(pd_val_frames)
        training_variables = self.properties['training-variables'].split(',')
        training_labels = self.properties['training-labels'].split(',')
        self.tot_events = pd_train.values.shape[0]
        self.properties['number-of-events'] = int(self.properties['number-of-events'])
        self.checkEventNumber()
        self.data_train = pd_train[training_variables].values[:self.properties['number-of-events']]
        self.properties['input_dim'] = len(training_variables)
        self.properties['output-dim'] = int(properties['output-dim'])
        self.labels_train = pd_train[training_labels].values[:self.properties['number-of-events']]
        self.data_val = pd_val[training_variables].values[:self.properties['number-of-events']]
        self.labels_val = pd_val[training_labels].values[:self.properties['number-of-events']]
        self.myconfig = config_file

    def readProperties(self):
        for section in self.config.sections():
            #remember to convert any config parameters that are not string
            for (key, val) in self.config.items(section):
                self.properties[key] = val

    def checkEventNumber(self):
        if self.properties['number-of-events'] > self.tot_events:
            raise ValueError('wrong number of events selected, the maximum number available for training is:' + str(self.tot_events))
        elif self.properties['number-of-events'] == -1 :
            self.properties['number-of-events'] = self.tot_events

    def gridTrain(self):
        if self.properties['unfold']: self.grid, self.grid_order = Grid.unfold(self.properties)
        else: self.grid, self.grid_order = Grid.concatenate(self.properties)

        base_name = self.properties['output-folder']
        if os.path.exists(base_name):
            raise ValueError('Error: folder '+base_name+' already exists')

        if(len(self.grid) > 0):
            os.system('mkdir ' + base_name)
            for parameters in self.grid:
                model_name = ""
                for i in range(len(parameters)):
                    print('parameters' + str(parameters[i]))
                    self.properties[self.grid_order[i]] = parameters[i]
                    model_name += self.grid_order[i][:3] + parameters[i]
                self.properties['output-folder'] = base_name + '/' + model_name
                self.trainDNN()
        else: self.trainDNN()
        copyfile(self.myconfig, base_name+ "/thisconfig.cfg")

    def trainDNN(self):
        os.system('mkdir ' + self.properties['output-folder'])
        #add this clear
        K.clear_session()
        #convert string to int
        self.properties['epochs'] = int(self.properties['epochs'])
        self.properties['batch-size'] = int(self.properties['batch-size'])
        
        #scaler implementation
        scaler = StandardScaler()
        scaler.fit(self.data_train)

        self.data_train_scaled = scaler.transform(self.data_train)
        self.data_val_scaled = scaler.transform(self.data_val)

        joblib.dump(scaler, self.properties['output-folder']+ "/scaler.pkl")

        if self.properties['scale-label'] == '1':
            label_scaler = StandardScaler()
            if self.properties['output-dim'] == 1:
                label_scaler.fit(np.expand_dims(self.labels_train,1))
                self.labels_train = label_scaler.transform(np.expand_dims(self.labels_train,1))
                self.labels_val = label_scaler.transform(np.expand_dims(self.labels_val,1))
            else:
                label_scaler.fit(self.labels_train)
                self.labels_train = label_scaler.transform(self.labels_train)
                self.labels_val = label_scaler.transform(self.labels_val)
                
            joblib.dump(label_scaler, self.properties['output-folder']+ "/label_scaler.pkl")

        model = Model.build(self.properties)

        if self.properties['save-steps']:
            print('I am here')
            print(self.properties['save-steps'])
            auto_save = ModelCheckpoint(self.properties['output-folder'] +"/current_model_epoch{epoch:02d}",monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
            print(auto_save)
        else:
            auto_save = ModelCheckpoint(self.properties['output-folder'] + "/current_model", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=2)

        history = model.fit(self.data_train_scaled, self.labels_train,
                        validation_data = (self.data_val_scaled, self.labels_val),
                        epochs=self.properties['epochs'],
                        batch_size=self.properties['batch-size'], shuffle=True,
                        callbacks=[auto_save])
                        #early stop to be implemented
        plot_history([('DNN model', history),],self.properties['output-folder'],self.properties['metrics'])
        
