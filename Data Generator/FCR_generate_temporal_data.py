import datetime as dt
import pandas as pd
import numpy as np
from keras.layers import Input, Dense, LSTM
from keras.models import Model
from keras import optimizers
from keras.losses import mean_squared_error
import scipy.io as spio


""" 
Script Description:

This script loads the datasets along with the temporal features extracted by the autoencoder and performs data
preprocessing as described in the Appendix. This script generates a datasets which is going to be used by the PGA-LSTM 
and other baselines. 

Note: ROA refers to the Falling Creek Reservoir (FCR) in Roanoke, Virginia.

"""



#loading the Datasets
dir = '../Datasets/'
glm_temp = pd.read_csv( dir + 'FCR_2013_2018_GLM_output.csv');
actual_temp = pd.read_csv(dir + 'FCR_2013_2018_Observed_with_GLM_output.csv');
time_features = pd.read_csv(dir + 'FCR_2013_2018_Drivers.csv');

#loading the extracted temporal features
filename = 'ROA_temporal_feature_autoencoder_lookback_7.mat'
matfile = spio.loadmat( dir + filename, squeeze_me = True,
                       variable_names = ['embedding_features'])
#lookback for auto-encoder
lookback = 7;
embedding_features = matfile['embedding_features'];

feature_mat = np.asarray(time_features.iloc[ : , 1 : ].values.tolist());
feature_mat = feature_mat[ lookback : , : ];

all_dates = time_features.iloc[ : , 0 ].tolist();
all_dates = all_dates[ lookback : ]
doy = np.zeros(feature_mat.shape[0]);

#convert list to datetime and computing the day of year feature
all_dates = [dt.datetime.strptime(x,'%Y-%m-%d') for x in all_dates]
for i in range(feature_mat.shape[0]):
    doy[i] = int(all_dates[i].strftime('%j'));

#extracts the GLM output features
glm_temp_features = np.asarray(glm_temp.iloc[ : , 1 : ].values.tolist())
glm_all_dates = glm_temp.iloc[ : , 0 ].tolist();
glm_all_dates = [dt.datetime.strptime(x,'%Y-%m-%d') for x in glm_all_dates]

#reshaping and extracting the temperature and depth features from GLM.
repeat = len(np.unique(glm_all_dates))
temp = [];
for i in np.arange((int)(glm_temp_features.shape[0]/repeat)):
    temp.append(glm_temp_features[ i * repeat : ( i + 1 ) * repeat])
temp = np.asarray(temp)
glm_temp_features = np.swapaxes(temp , 0 , 1);
glm_all_dates = glm_all_dates[ : repeat];


glm_temp_features = glm_temp_features[ 6 : , : , : ];
glm_all_dates = glm_all_dates[ 6 : ];

#computing the growing degree days features
year = [all_dates[i].year for i in range(len(all_dates))]
ymin = np.min(year);
ymax = np.max(year);
min_temp = 5;
growing_deg_days = np.zeros((feature_mat.shape[0]));
for y in range( ymin , ymax + 1):
    ix = (np.asarray(year) == y);
    year_mean = np.mean(glm_temp_features[ ix , : , 1 ]);
    current_temp = feature_mat[ ix , 2 ];
    current_temp[ current_temp < min_temp ] = min_temp;
    growing_deg_days[ix] = np.mean(current_temp) - min_temp;

#concatenating the features day of year, growing degree days and temporal embedding features to the feature set
feature_mat = np.column_stack((feature_mat, doy, growing_deg_days, embedding_features))
temp_feat = [];
depth_steps = glm_temp_features.shape[1]
for i in range(len(glm_all_dates)):
    temp_feat.append(np.column_stack((np.tile(feature_mat[ i , : ],( depth_steps , 1 )) , glm_temp_features[ i , : , : ])));
temp_feat = np.asarray(temp_feat);

#reshaping and extracting
actual_temp_mat = np.asarray(actual_temp.iloc[:,[ 1 , 3 ]].values.tolist());
actual_temp_mat = actual_temp_mat.reshape((int)(actual_temp_mat.shape[0] / depth_steps) , depth_steps , actual_temp_mat.shape[1])
actual_dates = actual_temp.iloc[ : , 0 ].tolist();
actual_dates = [dt.datetime.strptime(x,'%Y-%m-%d') for x in actual_dates]
date = [];
for i in range(len(actual_dates)):
    if i % depth_steps == 0:
        date.append(actual_dates[i]);

actual_dates = date;

#first 2 dates falls in the lookback region
actual_dates = actual_dates[ 2 : ];
actual_temp_mat = actual_temp_mat[ 2 : , : , : ];
X = [];
Y = [];
count = 0;
for i in range(len(glm_all_dates)):
    if count >= len(actual_dates):
        break;
    if actual_dates[count] == all_dates[i]:
        X.append(temp_feat[ i , : , : ]);
        Y.append(actual_temp_mat[ count , : , 1 ]);
        count = count + 1;
X = np.asarray(X);
Y = np.asarray(Y);

actual_year = np.asarray([actual_dates[i].year for i in range(len(actual_dates))])
split_year = 2017;
n_obs = (np.where(actual_year==split_year))[0][0] - 1;

y_mask = np.ones_like(Y);
y_new = np.zeros((Y.shape[0],Y.shape[1],2));
y_new[:,:,0] = Y;
y_new[:,:,1] = y_mask;

#creating train splits
train_X, train_Y = X[ : n_obs , : , : ], y_new[ : n_obs, : ];
train_dates = actual_dates[ : n_obs ];
train_year = [train_dates[i].year for i in range(train_X.shape[0])];
train_month = [train_dates[i].month for i in range(train_X.shape[0])];
train_day = [train_dates[i].day for i in range(train_X.shape[0])];

#creating test splits
test_X, test_Y = X[ n_obs : , : , : ], y_new[ n_obs : , : ];
test_dates = actual_dates[ n_obs : ];
test_year = [test_dates[i].year for i in range(test_X.shape[0])];
test_month = [test_dates[i].month for i in range(test_X.shape[0])];
test_day = [test_dates[i].day for i in range(test_X.shape[0])];

#saving FCR dataset
spio.savemat('../Datasets/ROA_temporal_mendota_train_test_split_4_year_train_new.mat',{'train_X':train_X,'train_Y_true':train_Y,'test_X':test_X,'test_Y_true':test_Y,
               'train_year':train_year,'train_month':train_month,'train_day':train_day,'test_year':test_year,'test_month':test_month,'test_day':test_day});




