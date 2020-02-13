# Author: T. Aarrestad, CERN
#
#
# ==============================================================================
"""Credits"""

# This code was adapted from
#
# https://github.com/jmduarte/JEDInet-code
#
# and takes advantage of the libraries at
#
# https://github.com/SheffieldML/GPy
# https://github.com/SheffieldML/GPyOpt


import sys
import h5py
import glob
import numpy as np
# keras imports
print("Importing TensorFlow")
import tensorflow
from tensorflow.keras.datasets import mnist, fashion_mnist
import matplotlib; matplotlib.use('PDF') 
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, Dropout, Flatten
from tensorflow.keras.layers import Concatenate, BatchNormalization, Activation
from tensorflow.keras.layers import MaxPooling2D, MaxPooling3D
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.utils import to_categorical

# Helper libraries
import GPyOpt
import GPy

nclasses=10
####################################################

# myModel class
class myModel():
    def __init__(self, optmizer_index=0, CNN_filters=10, 
                 CNN_filter_size=5, CNN_MaxPool_size=5, CNN_layers=2, CNN_activation_index=0, DNN_neurons=40, 
                 DNN_layers=2, DNN_activation_index=0, dropout=0.2, batch_size=100, epochs=50):  
       
        self.activation = ['relu', 'selu', 'elu']
        self.optimizer = ['adam', 'nadam','adadelta']
        self.optimizer_index = optmizer_index
        self.CNN_filters = CNN_filters
        self.CNN_filter_size = CNN_filter_size
        self.CNN_MaxPool_size = CNN_MaxPool_size
        self.CNN_layers = CNN_layers
        self.CNN_activation_index =  CNN_activation_index
        self.DNN_neurons = DNN_neurons
        self.DNN_layers = DNN_layers
        self.DNN_activation_index = DNN_activation_index
        self.dropout = dropout
        self.batch_size = batch_size
        # here an epoch is a single file
        self.epochs = epochs
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        X_test   = X_test.astype('float32')
        X_train  = X_train.astype('float32')
        X_test  /= 255
        X_train /= 255
        self.__y_test  = to_categorical(y_test, nclasses)
        self.__y_train = to_categorical(y_train, nclasses)
        self.__x_test  = np.expand_dims(X_test, axis=1)
        self.__x_train = np.expand_dims(X_train, axis=1)
        
        self.__model   = self.build()
    
    #  model
    def build(self):
        inputImage = Input(shape=(image_shape))
        x = Conv2D(self.CNN_filters, kernel_size=(self.CNN_filter_size,self.CNN_filter_size), 
                   data_format="channels_first", strides=(1, 1), padding="same", input_shape=image_shape,
                   kernel_initializer='lecun_uniform', name='cnn2D_0')(inputImage) #he_uniform
        x = BatchNormalization()(x)
        x = Activation(self.activation[self.CNN_activation_index])(x)
        x = MaxPooling2D( pool_size = (self.CNN_MaxPool_size,self.CNN_MaxPool_size))(x)
        x = Dropout(self.dropout)(x)
        for i in range(1,self.CNN_layers):
            x = Conv2D(self.CNN_filters, kernel_size=(self.CNN_filter_size,self.CNN_filter_size), 
                   data_format="channels_first", strides=(1, 1), padding="same", input_shape=image_shape,
                    kernel_initializer='lecun_uniform', name='cnn2D_%i' %i)(x)
            x = BatchNormalization()(x)
            x = Activation(self.activation[self.CNN_activation_index])(x)
            x = MaxPooling2D( pool_size = (self.CNN_MaxPool_size,self.CNN_MaxPool_size))(x)
            x = Dropout(self.dropout)(x)
            
        ####
        x = Flatten()(x)
        #
        for i in range(self.DNN_layers):
            x = Dense(self.DNN_neurons, activation=self.activation[self.DNN_activation_index], 
                      kernel_initializer='lecun_uniform', name='dense_%i' %i)(x)
            x = Dropout(self.dropout)(x)
       
        output = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform', 
                       name = 'output_softmax')(x)
        ####
        model = Model(inputs=inputImage, outputs=output)
        model.compile(optimizer=self.optimizer[self.optimizer_index], 
                      loss='categorical_crossentropy', metrics=['acc'])
        return model

    
    # fit model
    def model_fit(self):

        self.__model.fit(self.__x_train, self.__y_train, epochs=self.epochs, 
                                   batch_size= self.batch_size, validation_data=[self.__x_test, self.__y_test],verbose=0, 
                                   callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0),
                                                           ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0), 
                                                           TerminateOnNaN()])
    # evaluate  model
    def model_evaluate(self):
        self.model_fit()
        evaluation = self.__model.evaluate(self.__x_test, self.__y_test, batch_size=self.batch_size, verbose=0)
        return evaluation


####################################################

# Runner function for model
# function to run  class

def run_model(optmizer_index=0, CNN_filters=10, 
              CNN_filter_size=5, CNN_MaxPool_size=2, CNN_layers=2, CNN_activation_index=0, DNN_neurons=40, 
              DNN_layers=2, DNN_activation_index=0, dropout=0.2, batch_size=100, epochs=50):
    
    _model = myModel( optmizer_index, CNN_filters, CNN_MaxPool_size, CNN_filter_size,
                 CNN_layers, CNN_activation_index, DNN_neurons, DNN_layers, DNN_activation_index, 
                 dropout, batch_size, epochs)
    model_evaluation = _model.model_evaluate()
    return model_evaluation

n_epochs = 20

image_shape = (1, 28, 28)

# Bayesian Optimization

# the bounds dict should be in order of continuous type and then discrete type
bounds = [{'name': 'optmizer_index',        'type': 'discrete',   'domain': (0,1)},
          {'name': 'CNN_filters',           'type': 'discrete',   'domain': (8, 16, 32, 64)},
          {'name': 'CNN_filter_size',       'type': 'discrete',   'domain': (2, 3)},
          {'name': 'CNN_MaxPool_size',      'type': 'discrete',   'domain': (2, 3)},
          {'name': 'CNN_layers',            'type': 'discrete',   'domain': (2, 3)},
          {'name': 'CNN_activation_index',  'type': 'discrete',   'domain': (0, 1)},
          {'name': 'DNN_neurons',           'type': 'discrete',   'domain': (28, 64, 128)},
          {'name': 'DNN_layers',            'type': 'discrete',   'domain': (1, 2, 3)},
          {'name': 'DNN_activation_index',  'type': 'discrete',   'domain': (0, 1)},
          {'name': 'dropout',               'type': 'continuous', 'domain': (0.25, 0.5)},
          {'name': 'batch_size',            'type': 'discrete',   'domain': (32, 50)}]

# function to optimize model
def f(x):
    print "x parameters are"
    print(x)
    evaluation = run_model(optmizer_index       = int(x[:,0]), 
                           CNN_filters          = int(x[:,1]), 
                           CNN_filter_size      = int(x[:,2]),
                           CNN_MaxPool_size     = int(x[:,3]),
                           CNN_layers           = int(x[:,4]), 
                           CNN_activation_index = int(x[:,5]), 
                           DNN_neurons          = int(x[:,6]), 
                           DNN_layers           = int(x[:,7]),
                           DNN_activation_index = int(x[:,8]),
                           dropout              = float(x[:,9]),
                           batch_size           = int(x[:,10]),
                           epochs = n_epochs)
    print("LOSS:\t{0} \t ACCURACY:\t{1}".format(evaluation[0], evaluation[1]))
    print(evaluation)
    return evaluation[0]

if __name__ == "__main__":
  opt_model = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)
  opt_model.run_optimization(max_iter=10)

  print("DONE")
  print("x:",opt_model.x_opt)
  print("y:",opt_model.fx_opt)

  # print optimized model
  print("""
  Optimized Parameters:
  \t{0}:\t{1}
  \t{2}:\t{3}
  \t{4}:\t{5}
  \t{6}:\t{7}
  \t{8}:\t{9}
  \t{10}:\t{11}
  \t{12}:\t{13}
  \t{14}:\t{15}
  \t{16}:\t{17}
  \t{18}:\t{19}
  \t{20}:\t{21}
  """.format(bounds[0]["name"],opt_model.x_opt[0],
             bounds[1]["name"],opt_model.x_opt[1],
             bounds[2]["name"],opt_model.x_opt[2],
             bounds[3]["name"],opt_model.x_opt[3],
             bounds[4]["name"],opt_model.x_opt[4],
             bounds[5]["name"],opt_model.x_opt[5],
             bounds[6]["name"],opt_model.x_opt[6],
             bounds[7]["name"],opt_model.x_opt[7],
             bounds[8]["name"],opt_model.x_opt[8],
             bounds[9]["name"],opt_model.x_opt[9],
             bounds[10]["name"],opt_model.x_opt[10]))
  print("optimized loss: {0}".format(opt_model.fx_opt))