from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
#saving and loading
import scipy.io as spio

#visualization
from matplotlib import pyplot
from matplotlib.colors import LogNorm

import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Input, Dense

from keras.layers import Masking
from keras.layers import Dropout
from keras import backend as K
from keras.layers import Bidirectional, TimeDistributed, LSTM
from keras.callbacks import EarlyStopping, TerminateOnNaN
from keras.models import Model
from keras import optimizers
import h5py
from keras.layers import concatenate
from keras import regularizers
import keras
import random
import tensorflow as tf
from keras.engine import InputSpec
from scipy import stats
import os


# %% Importing Rectifier LSTM
import sys
sys.path.append("../Monotonic LSTM Script")
from monotonic_lstm_2denselayer import mLSTM

# %% Dropout Class
""" This class is used to generate K samples during the testing time using MC Dropout.
    Implements the MC dropout in the paper : "Dropout as a Bayesian Approximation : Representing Model Uncertainty in 
    Deep Learning"
"""
class KerasDropoutPrediction(object):
    def __init__(self ,model):
        self.f = K.function(
            [model.layers[0].input,
             K.learning_phase()],
            [model.layers[-1].output])
    def predict(self ,x, n_iter=10):
        result = []
        for _ in range(n_iter):
            result.append(self.f([x , 1]))
        result = np.array(result).reshape(n_iter ,x.shape[0] ,x.shape[1]).T
        return result



# %% Custom Loss functions

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

#loss functions with masking 
def masked_mean_squared_error(y_true ,y_pred):
    return K.mean(y_true[:, :, 1] * K.square(y_pred[:, :, 0] - y_true[:, :, 0]), axis=-1)


def masked_root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(y_true[:, :, 1] * K.square(y_pred[:, :, 0] - y_true[:, :, 0]), axis=-1))
  

# %% Density Conversion Functions
def density(temp):
    return (1 + (1 - (temp + 288.9414) * (temp - 3.9863) ** 2 / (508929.2 * (temp + 68.12963))))
    # [math.log(0.1+y) for y in x]
    # return y;

def actual_density(temp):
    return (1 - (temp + 288.9414) * (temp - 3.9863) ** 2 / (508929.2 * (temp + 68.12963)))



# %% Read Data Function

def read_Data():
  
  dir = '../Datasets/';
  if use_temporal_feature == 1:
      filename = 'temporal_mendota_train_test_split_4_year_train_new.mat';
  else:
      filename = 'new_data_mendota_train_test_split_4_year_train.mat';
  matfile = spio.loadmat(dir + filename, squeeze_me=True,
                         variable_names=['train_X','train_Y_glm','train_Y_true','test_X','test_Y_glm','test_Y_true'])

  train_X = matfile['train_X'];
  train_y = matfile['train_Y_true'];
  train_y_glm = matfile['train_Y_glm'];

  test_X = matfile['test_X'];
  test_y = matfile['test_Y_true'];
  test_y_glm = matfile['test_Y_glm'];
  params = [train_X, train_y, train_y_glm, test_X, test_y, test_y_glm];
  return params;

 # %% Create Train Data Subset
def train_subset(train_X, train_y, train_y_glm):
  
  count = np.zeros(train_y.shape[0]);
  for i in range(train_y.shape[0]):
      count[i] = np.count_nonzero(train_y[i,:,1]);
  tot_size = np.sum(count);  
  tr_size = tr_frac * tot_size / 100;
  index = (np.arange(train_y.shape[0])).tolist();
  ix = [];
  size = 0;
  while len(index) > 0:
      temp = index.pop(random.randrange( 0 , len(index) ));
      size = size + count[temp];
      ix.append(temp);
      if size >= tr_size:
          break;
  ix = np.sort(np.asarray(ix));
  train_X = train_X[ix];
  train_y = train_y[ix];
  train_y_glm = train_y_glm[ix];
  params = [ix, train_X, train_y, train_y_glm];
  return params;

# %% Matrix 2d to 3d and vice versa functions

def transform_3d_to_2d(X):
  return X.reshape(( X.shape[0] * X.shape[1], X.shape[2] ));


def transform_2d_to_3d(X, steps):
  return X.reshape((int(X.shape[0] / steps), steps, X.shape[1]));

# %% Masking output Label function
def masking_output_labels(
    y_density,
    y_temperature,
    mask_value):
  
  
  for i in range(y_density.shape[0]):
        if math.isnan(y_density[i, 0]):
            y_density[i, 0] = mask_value;
            y_temperature[i, 0] = mask_value;
  params = [y_density, y_temperature];
  return params;

# %% Edge padding functions
def edge_padding(
    X,
    y_density,
    y_temperature,
    pad_steps):
  
  
  X_pad = np.zeros((X.shape[0], X.shape[1] + pad_steps, X.shape[2]))
  y_density_pad = np.zeros((y_density.shape[0], y_density.shape[1] + pad_steps, y_density.shape[2]))
  y_temperature_pad = np.zeros((y_temperature.shape[0], y_temperature.shape[1] + pad_steps, y_temperature.shape[2]))
  for i in range(X.shape[0]):
      X_pad[i, :, :] = np.pad(X[i, :, :], ((pad_steps, 0), (0, 0)), 'edge')
      y_density_pad[i, :, :] = np.pad(y_density[i, :, :], ((pad_steps, 0), (0, 0)), 'edge')
      y_temperature_pad[i, :, :] = np.pad(y_temperature[i, :, :], ((pad_steps, 0), (0, 0)), 'edge')
  params=[X_pad, y_density_pad, y_temperature_pad];
  return params;

# %% Create Model for Temperature Prediction
def createModel(
        input_shape1,
        input_shape2,
        lstm_nodes, 
        lstm_bias, 
        drop_frac, 
        feedforward_nodes, 
        lamda_reg, 
        n_nodes):
  

  main_input = Input(shape = (input_shape1, input_shape2), name='main_input')
  mlstm_out = mLSTM(input_units = lstm_nodes, hidden_units = feedforward_nodes, output_units = 1, return_sequences = True, 
                      use_bias = lstm_bias, dropout = drop_frac, recurrent_dropout = drop_frac, hidden_dropout = drop_frac, name='aux_output')(main_input)
  x = concatenate([mlstm_out, main_input])
  dense3 = TimeDistributed(Dense(n_nodes, use_bias = 1,
               kernel_regularizer = regularizers.l2(lamda_reg)))(x)
  activ_dense3 = keras.layers.ELU(alpha=1.0)(dense3)
  main_output = Dense(1, activation = 'linear', name = 'main_output')(activ_dense3)
  
  model = Model(inputs = [main_input], outputs = [main_output, mlstm_out])
  return model;

# %% Train / Test RMSE prediction without dropout 
def normal_prediction( model, test_X, test_y , pad_steps):
  
  test_pred = model.predict({'main_input':test_X})
  test_pred = test_pred[0];
  test_pred = test_pred[ : , pad_steps : , : ];
  test_y = test_y[ : , pad_steps : , : ];
  test_rmse = np.mean(np.sqrt(np.divide(np.sum(np.multiply(test_y[ : , : , 1 ],
                                                    np.square(test_pred[ : , : , 0 ] - test_y[ : , : , 0 ])), axis = 1),
                                       np.sum(test_y[ : , : , 1 ], axis = 1))))
  params = [test_rmse, test_pred, test_y];
  return params;


# %% Train / Test RMSE prediction with dropout
def dropout_prediction( model, n_iter, test_X, test_y , pad_steps):
  
  kdp = KerasDropoutPrediction(model);
  test_pred = kdp.predict(test_X, n_iter = n_iter)
  test_pred = test_pred[ pad_steps : , : , : ]
  
  test_pred_uq_mean = test_pred.mean(axis = -1 ).transpose()
  test_pred_uq_std = test_pred.std(axis = -1 ).transpose()
  test_rmse_dropout=np.mean(np.sqrt(np.divide(np.sum(np.multiply(test_y[ : , : , 1 ],
                                                                 np.square(test_pred_uq_mean[ : , : ] - test_y[ : , : , 0 ])),axis = 1),
                                              np.sum(test_y[ : , : , 1 ], axis = 1))));
  params = [test_rmse_dropout, test_pred, test_y];
  return params;

# %% Compute Physical Inconsistency     
def normal_physical_inconsistency(tol, test_pred, depth_steps):
  
  test_density = actual_density(test_pred);
  test_count = np.zeros(test_density.shape[0]);
  for i in range(test_density.shape[0]):
      for j in range(test_density.shape[1] - 1):
          if test_density[ i , j ] - test_density[ i , j + 1 ] > tol:
              test_count[i] = test_count[i] + 1;
      test_count[i] = test_count[i] / depth_steps;

  test_incon = np.sum(test_count) / test_density.shape[0];
  return test_incon;

# %% Compute Physical Inconsistencies for all samples
def physical_inconsistency_all_sample(tol, test_pred_do, depth_steps):
  
  test_pred_do = np.swapaxes(test_pred_do,0,2);
  density_test_pred_do = actual_density(test_pred_do);
  
  test_count_uq = np.zeros(test_pred_do.shape[0]);
  for k in range(density_test_pred_do.shape[0]):
      test_count = np.zeros(density_test_pred_do.shape[1])
      for i in range(density_test_pred_do.shape[1]):
          for j in range(density_test_pred_do.shape[2] - 1):
              if density_test_pred_do[k,i,j] - density_test_pred_do[k,i,j + 1] > tol:
                  test_count[i] = test_count[i] + 1;
          test_count[i] = test_count[i] / depth_steps;
      test_count_uq[k] = np.sum(test_count) / density_test_pred_do.shape[1];
  test_incon_uq = np.sum(test_count_uq) / density_test_pred_do.shape[0];


  test_incon_uq = np.sum(test_count_uq) / density_test_pred_do.shape[0];
  return test_incon_uq;



# %% Start of PGA model function
#Start of function
def PGA_mLSTM_train_test(iteration, 
                         tr_frac, 
                         val_frac, 
                         patience_val, 
                         num_epochs, 
                         batch_size, 
                         lstm_nodes, 
                         feedforward_nodes, 
                         mask_value, 
                         drop_frac, 
                         lstm_bias, 
                         n_nodes, 
                         use_GLM, 
                         use_temporal_feature, 
                         lamda_reg, 
                         shuffle, 
                         lamda_main, 
                         lamda_aux, 
                         tol, 
                         pad_steps):
    
    
    #read Data
    [train_X,train_y_temperature,train_y_glm,test_X,test_y_temperature,test_y_glm] = read_Data();
    
    #Add GLM feature
    if use_GLM == 1:
        train_X = np.dstack((train_X, train_y_glm));
        test_X = np.dstack((test_X, test_y_glm));
    
    depth_steps = train_X.shape[1];
    
    #Normalise Data
    train_X = transform_3d_to_2d(train_X);
    test_X = transform_3d_to_2d(test_X);
    
    m1_train_mean = train_X.mean(axis=0);
    m1_train_std = train_X.std(axis=0);
    train_X = (train_X - m1_train_mean) / m1_train_std;
    test_X = (test_X - m1_train_mean) / m1_train_std;
    
    train_X = transform_2d_to_3d(train_X, depth_steps);
    test_X = transform_2d_to_3d(test_X, depth_steps);
    
    #Create train subset
    [ix, train_X, train_y_temperature, train_y_glm] = train_subset(train_X, train_y_temperature, train_y_glm);
    
    #creating path and filename for storage of results
    exp_name = 'pga_lstm_model_' + str(num_epochs)  + '_dropout_frac_' + str(drop_frac) + '_feedforward_nodes' + str(feedforward_nodes) + '_lstm_nodes' + str(
        lstm_nodes) + '_val_frac' + str(val_frac) + '_trfrac' + str(tr_frac) + '_iter' + str(iteration)+'_n_nodes'+str(n_nodes)+'_use_GLM'+str(use_GLM)+ \
            '_use_temporal_features'+str(use_temporal_feature) + '_lamda_reg'+str(lamda_reg)+'_pad_steps'+str(pad_steps)+'_layers_in_rec_lstm'+str(layers_in_rec_lstm)
    exp_name = exp_name.replace('.', 'pt')
    
    results_dir = '../Results/Mendota_PGA_mLSTM/'
    model_name = results_dir + exp_name + '_model.h5'  # storing the trained model
    results_name = results_dir + exp_name + '_results.mat'  # storing the results of the model
    
    train_y_temperature = transform_3d_to_2d(train_y_temperature);
    test_y_temperature = transform_3d_to_2d(test_y_temperature);
    
    #Create Density labels 
    train_y_density = np.zeros(np.shape(train_y_temperature));
    test_y_density = np.zeros(np.shape(test_y_temperature));
    train_y_density[:,1], test_y_density[:,1] = train_y_temperature[:,1], test_y_temperature[:,1]
    train_y_density[:,0], test_y_density[:,0] = density(train_y_temperature[:,0]), density(test_y_temperature[:,0])
    
    # density output normalisation
    mean = np.nanmean(train_y_density[:, 0]);
    std = np.nanstd(train_y_density[:, 0]);
    train_y_density[:, 0] = (train_y_density[:, 0] - mean) / std;
    test_y_density[:, 0] = (test_y_density[:, 0] - mean) / std;
    
    #masking output labels
    [train_y_density, train_y_temperature] = masking_output_labels(train_y_density, train_y_temperature, mask_value);
    [test_y_density, test_y_temperature] = masking_output_labels(test_y_density, test_y_temperature, mask_value);
    
    #reshaping train/test output density/temperature labels
    train_y_density = transform_2d_to_3d(train_y_density, depth_steps); 
    train_y_temperature = transform_2d_to_3d(train_y_temperature, depth_steps); 
    test_y_density = transform_2d_to_3d(test_y_density, depth_steps); 
    test_y_temperature = transform_2d_to_3d(test_y_temperature, depth_steps); 
    
    #perform edge padding
    [train_X_pad,train_y_density_pad,train_y_temperature_pad] = edge_padding(train_X,train_y_density,train_y_temperature,pad_steps);
    [test_X_pad,test_y_density_pad,test_y_temperature_pad] = edge_padding(test_X,test_y_density,test_y_temperature,pad_steps);
    
    if shuffle == 1:
        ix = np.arange(train_X_pad.shape[0]);
        random.shuffle(ix);
        train_X_pad = train_X_pad[ix,:,:];
        train_y_density_pad = train_y_density_pad[ix,:,:];
        train_y_temperature_pad = train_y_temperature_pad[ix,:,:];
    
    
    #create auxiliary dataset (same as input data)
    aux_train_X = train_X_pad;
    aux_test_X = test_X_pad;
    
    model=createModel(
        train_X_pad.shape[1],
        train_X_pad.shape[2],
        lstm_nodes, 
        lstm_bias, 
        drop_frac, 
        feedforward_nodes, 
        lamda_reg, 
        n_nodes);
    
    model.summary();
    
    #optimiser = optimizers.Adadelta(clipnorm=1)
    optimiser = optimizers.Adam(clipnorm=1)
    
    model.compile(
        loss={'main_output': masked_mean_squared_error, 'aux_output': masked_mean_squared_error}, 
        optimizer = optimiser,
        loss_weights = {'main_output':lamda_main,'aux_output':lamda_aux})
    
    
    early_stopping = EarlyStopping(
                            monitor = 'val_main_output_loss', 
                            patience = patience_val, 
                            verbose = 1)
    history=model.fit(
                {'main_input':train_X_pad},
                {'main_output':train_y_temperature_pad,'aux_output':train_y_density_pad},
                epochs = num_epochs, 
                batch_size = batch_size, 
                verbose = 2, 
                shuffle = False,
                validation_split = val_frac, 
                callbacks = [early_stopping, TerminateOnNaN()]);
    
    pyplot.plot(history.history['loss'], label='train_loss')
    pyplot.plot(history.history['main_output_loss'], label='main_output_loss')
    pyplot.plot(history.history['aux_output_loss'], label='aux_output_loss')
    pyplot.plot(history.history['val_loss'], label='val_loss')
    pyplot.plot(history.history['val_main_output_loss'], label='val_main_output_loss')
    pyplot.plot(history.history['val_aux_output_loss'], label='val_aux_output_loss')
    pyplot.legend()
    pyplot.show()
            
    #Calculating model RMSE 
    [test_rmse1, test_pred1, test_y1] = normal_prediction(model, test_X_pad, test_y_temperature_pad, pad_steps);
    [train_rmse1, train_pred1, train_y1] = normal_prediction(model, train_X_pad, train_y_temperature_pad, pad_steps);
    
    print('Without dropout = TrainRMSE : ' +str(train_rmse1) + ' TestRMSE : '+str(test_rmse1));
    
    [test_rmse_dropout, test_pred_do, test_y1] = dropout_prediction(model,100,test_X_pad, test_y_temperature_pad[:,pad_steps:,:], pad_steps);
    [train_rmse_dropout, train_pred_do, train_y1] = dropout_prediction(model,100,train_X_pad, train_y_temperature_pad[:,pad_steps:,:], pad_steps);
    
    print('With dropout = TrainRMSE : ' +str(train_rmse_dropout) + ' TestRMSE : '+str(test_rmse_dropout));
    
    
    #Calculating Physical Inconsistencies
    test_inconsistency_without_dropout = normal_physical_inconsistency(tol, test_pred1, depth_steps);
    train_inconsistency_without_dropout = normal_physical_inconsistency(tol, train_pred1, depth_steps);
    
    print("Without dropout : Test Incon = "+str(test_inconsistency_without_dropout)+'  Train Incon = '+str(train_inconsistency_without_dropout))
    
    test_pred_uq_mean = test_pred_do.mean(axis=-1).transpose();
    train_pred_uq_mean = train_pred_do.mean(axis=-1).transpose();
    test_inconsistency_dropout_mean = normal_physical_inconsistency(tol, test_pred_uq_mean, depth_steps);
    train_inconsistency_dropout_mean = normal_physical_inconsistency(tol, train_pred_uq_mean, depth_steps);
    
    print("With dropout Inconsistency of sample mean: Test Incon = "+str(test_inconsistency_dropout_mean)+'  Train Incon = '+str(train_inconsistency_dropout_mean))
    
    test_inconsistency_dropout_all = physical_inconsistency_all_sample(tol, test_pred_do, depth_steps);
    train_inconsistency_dropout_all = physical_inconsistency_all_sample(tol, train_pred_do, depth_steps);
    
    print("With dropout Inconsistency of all samples: Test Incon = "+str(test_inconsistency_dropout_all)+'  Train Incon = '+str(train_inconsistency_dropout_all))
    
    
    #model.save(model_name)
    
    spio.savemat(results_name,{'test_pred_temperature':test_pred1,
                               'test_y_temperature':test_y1,
                               'train_pred_temperature':train_pred1, 
                               'train_y_temperature':train_y1,
                               'train_rmse':train_rmse1,
                               'test_rmse':test_rmse1,
                               'train_rmse_dropout':train_rmse_dropout,
                               'test_rmse_dropout':test_rmse_dropout,
                               'test_incon_without_dropout':test_inconsistency_without_dropout,
                               'train_inconsistency_without_dropout':train_inconsistency_without_dropout,
                               'test_inconsistency_dropout_mean':test_inconsistency_dropout_mean,
                               'train_inconsistency_dropout_mean':train_inconsistency_dropout_mean,
                               'test_inconsistency_dropout_all':test_inconsistency_dropout_all,
                               'train_inconsistency_dropout_all':train_inconsistency_dropout_all,
                               'test_predictions_all':test_pred_do,
                               'train_predictions_all':train_pred_do,
                               'train_index':ix,
                               'train_main_loss':history.history['main_output_loss'],
                               'train_aux_output_loss':history.history['aux_output_loss'],
                               'train_loss':history.history['loss'], 
                               'train_val_loss':history.history['val_loss']
                               })
    
    
# %% Main Function
if __name__ == '__main__':
    #set Parameters of model here    
    tr_frac_range = [10,20,30,40,50,100];
    tr_frac = tr_frac_range[3]; # 40% training fraction used
    layers_in_rec_lstm = 2;
    val_frac = 0.1;
    patience_val = 1000;
    num_epochs = 5000;
    batch_size = 20;
    lstm_nodes = 8;
    feedforward_nodes = 5;
    mask_value = 0;
    drop_frac = 0.2;
    lstm_bias = 1;
    n_nodes = 5;
    use_GLM = 1;
    use_temporal_feature = 1;
    lamda_reg = 0.05;
    iteration = 1;
    shuffle = 1;
    lamda_main = 0.2;
    lamda_aux = 1;
    tol=0.00001;
    pad_steps = 10;
    iteration = 1;
    PGA_mLSTM_train_test(iteration, tr_frac, val_frac, patience_val, num_epochs, batch_size, lstm_nodes, feedforward_nodes, mask_value, drop_frac, lstm_bias, 
                            n_nodes, use_GLM, use_temporal_feature, lamda_reg, shuffle, lamda_main, lamda_aux, tol, pad_steps)
