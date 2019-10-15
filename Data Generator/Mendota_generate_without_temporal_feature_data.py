import feather
import datetime
import numpy as np
from keras.layers import Input, Dense, LSTM
from keras.models import Model
from keras import optimizers
from keras.losses import mean_squared_error
import scipy.io as spio


""" 
Script Description:
This script loads the datasets and performs data preprocessing as described in the Appendix. 
This script generates a dataset (without temporal features) which is going to be used by the PGA-LSTM and other baselines. 

This datasets are for Lake Mendota
"""


#loading the Datasets
dir = '../Datasets/';
glm_temp = feather.read_dataframe(dir + 'mendota_GLM_uncal_temperatures_anuj.feather');
actual_temp = feather.read_dataframe(dir + 'Mendota_buoy_data_anuj.feather');
time_features = feather.read_dataframe(dir + 'mendota_meteo.feather');


#extracting the data from the dataframe
feature_mat = np.asarray(time_features.iloc[:,[1,2,3,4,5,6,7]].values.tolist());
glm_temp_mat = np.asarray(glm_temp.iloc[:,1:-1].values.tolist());
freeze = np.asarray(glm_temp.iloc[:,-1].values.tolist());
all_dates = time_features.iloc[:,0].values.tolist();
doy = np.zeros(feature_mat.shape[0]);


#convert list to datetime and computing the day of year feature
actual_temp_mat = np.asarray(actual_temp.iloc[:,:].values.tolist());
for i in range(feature_mat.shape[0]):
    doy[i] = int(all_dates[i].strftime('%j'));

#computing the growing degree days features
year = [all_dates[i].year for i in range(len(all_dates))]
ymin = np.min(year);
ymax = np.max(year);
min_temp = 5;
growing_deg_days = np.zeros((feature_mat.shape[0]));
for y in range( ymin , ymax + 1 ):
    ix = (np.asarray(year) == y);
    year_mean = np.mean(glm_temp_mat[ ix , : ]);
    current_temp = feature_mat[ ix , 2 ];
    current_temp[ current_temp < min_temp ] = min_temp;
    growing_deg_days[ix] = np.mean(current_temp) - min_temp;


#concatenating the features day of year and growing degree days to the feature set  
feature_mat = np.column_stack((feature_mat, doy, freeze, growing_deg_days));
n_features = feature_mat.shape[1];
depth_steps = glm_temp_mat.shape[1];
GLM_mat = np.zeros((feature_mat.shape[0], depth_steps, n_features + 2 ))

#creating separate matrix for GLM values
for i in range(feature_mat.shape[0]):
    depth = 0;
    for j in range(depth_steps):
        GLM_mat[ i , j , : -2 ] = feature_mat[ i , : ];
        GLM_mat[ i , j , -2 ] = depth;
        depth = depth + 0.5;
        GLM_mat[ i , j , -1 ] = glm_temp_mat[ i , j ];

#merging the features and the output labels with discretization
final = np.zeros(( feature_mat.shape[0], depth_steps , n_features + 3 ))
current_date = actual_temp_mat[ 0 , 0];
count = 0;
for i in range(feature_mat.shape[0]):
    final[ i , : , : -1 ] = GLM_mat[ i , : , : ];
    final[ i , : , -1 ] = float('nan');
    if all_dates[i] == current_date:
        while( current_date == actual_temp_mat[ count , 0 ]):
           depth = 0;
           for j in range(depth_steps):
               if( depth == actual_temp_mat[ count , 1 ]):
                   final[ i , j , -1 ] = actual_temp_mat[ count , 2 ];
               depth = depth + 0.5;
           count = count + 1;
           if count == actual_temp_mat.shape[0]:
              break;
        if count == actual_temp_mat.shape[0]:
              break;      
        current_date = actual_temp_mat[ count , 0 ];
        
X = final[ : , : , : -2 ];
Y_glm = final[ : , : , -2 ];
Y_true = final[ : , : , -1 ];        

mask = ~np.isnan(Y_true)
Y_true = np.dstack(( Y_true , mask));

#train_test_split
start_date = all_dates[0];
count_years = 0;
train_data_years = 4;
for i in range( 1 , feature_mat.shape[0]):
    if start_date.month == all_dates[i].month and start_date.day == all_dates[i].day:
        count_years = count_years + 1;
    if train_data_years == count_years:
        break;
n_obs = i;

train_X, train_Y_glm, train_Y_true = X[ : n_obs , : , : ] , Y_glm[ : n_obs , : ], Y_true[ : n_obs , : ];
test_X, test_Y_glm, test_Y_true = X[ n_obs : , : , : ], Y_glm[ n_obs : , : ], Y_true[ n_obs : , : ];
date_train, date_test = all_dates[ : n_obs ] , all_dates[ n_obs : ];

#remove no real observation days.
train_y_temp = [];
train_X_temp = [];
train_y_glm_temp = [];
train_ix = np.zeros(0);
train_date = [];
for i in range(train_Y_true.shape[0]):
    if np.count_nonzero(train_Y_true[i, :, 1]) != 0:
        train_X_temp.append(train_X[i, :, :]);
        train_y_temp.append(train_Y_true[i, :, :]);
        train_y_glm_temp.append(train_Y_glm[i, :]);
        train_ix = np.append(train_ix, i);
        train_date.append(date_train[i]);

train_Y_true = np.asarray(train_y_temp);
train_X = np.asarray(train_X_temp);
train_Y_glm = np.asarray(train_y_glm_temp);
train_year = [train_date[i].year for i in range(train_X.shape[0])];
train_month = [train_date[i].month for i in range(train_X.shape[0])];
train_day = [train_date[i].day for i in range(train_X.shape[0])];

test_y_temp = [];
test_X_temp = [];
test_y_glm_temp = [];
test_ix = np.zeros(0);
test_date = [];
for i in range(test_Y_true.shape[0]):
    if np.count_nonzero(test_Y_true[i, :, 1]) != 0:
        test_X_temp.append(test_X[i, :, :]);
        test_y_temp.append(test_Y_true[i, :, :]);
        test_y_glm_temp.append(test_Y_glm[i, :]);
        test_ix = np.append(test_ix,i);
        test_date.append(date_test[i]);

test_Y_true = np.asarray(test_y_temp);
test_X = np.asarray(test_X_temp);
test_Y_glm = np.asarray(test_y_glm_temp);
test_year = [test_date[i].year for i in range(test_X.shape[0])];
test_month = [test_date[i].month for i in range(test_X.shape[0])];
test_day = [test_date[i].day for i in range(test_X.shape[0])];


spio.savemat('../Datasets/new_data_mendota_train_test_split_4_year_train.mat',{'train_X':train_X,'train_Y_glm':train_Y_glm,'train_Y_true':train_Y_true,'test_X':test_X,'test_Y_glm':test_Y_glm,'test_Y_true':test_Y_true, 
                                                                   'train_year':train_year,'train_month':train_month,'train_day':train_day,'test_year':test_year,'test_month':test_month,'test_day':test_day});

           
