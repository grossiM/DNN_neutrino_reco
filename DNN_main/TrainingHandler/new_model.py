import sys

from keras.models import Sequential, load_model
from keras.activations import relu
from keras.layers import Dense, LeakyReLU, Dropout
from keras.optimizers import Adam, SGD
from keras import metrics
from keras import losses
import keras.backend as K
import numpy as np 

def build(properties):

    if(properties['model'] == 'custom_model'):
        if not 'hidden-layers' in properties:
            raise ValueError('Error: Property hidden-layers is required in the section training')
        if not 'neurons' in properties:
            raise ValueError('Error: Property neurons is required in the section training')
        if not 'dropout-rate' in properties:
            raise ValueError('Error: Property dropout-rate is required in the section training')
        model = Sequential()

        properties['hidden-layers'] = int(properties['hidden-layers'])
        properties['neurons'] = int(properties['neurons'])
        properties['first-neuron'] = int(properties['first-neuron'])
        properties['last-neuron'] = int(properties['last-neuron'])
        properties['dropout-rate'] = float(properties['dropout-rate'])

    
        if properties['hidden-layers'] > 0:
            model.add(Dense(units=properties['neurons'], input_dim=properties['input_dim'],
                            kernel_initializer='normal', activation="relu"))

        dropout=True

        if (properties['dropout-rate'] < 1e-8) : dropout=False

        for i in range(properties['hidden-layers'] - 1):

            if dropout : model.add(Dropout(properties['dropout-rate']))
            #create new model name and separate shape for grid search
            if properties['shape'] == 'funnel':
                n = properties['first-neuron']
                n -= ((properties['first-neuron'] - properties['last-neuron']) / properties['hidden-layers'])
                print(n)
            elif properties['shape'] == 'triangle':
                nt = np.linspace(properties['first-neuron'],properties['last-neuron'],properties['hidden-layers']+2,dtype=int).tolist()
                nt.reverse()
                n = nt.pop(i)
            else: 
                print('ERROR')
                sys.exit(0)
                
            model.add(Dense(units=n,
                            kernel_initializer=properties['kernel_init'],
                            activation=properties['activation']))

        if dropout : model.add(Dropout(properties['dropout-rate']))
        model.add(Dense(units=1, 
                        kernel_initializer=properties['kernel_init'],
                        activation=properties['last_activation']))

        model.compile(loss=properties['loss'],
                        optimizer=properties['optimizer'],
                        metrics=[properties['metrics']])
        return model
