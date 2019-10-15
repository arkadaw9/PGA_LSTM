import feather
import numpy as np
from keras.layers import Input, Dense, LSTM, RepeatVector
from keras.models import Model
from keras import optimizers
from keras.losses import mean_squared_error
import scipy.io as spio

"""
Script Description:
This script traing the LSTM based autoencoder to extract temporal features from the Lake Mendota data.
"""

dir = '../Datasets/';
glm_temp = feather.read_dataframe(dir+'mendota_GLM_uncal_temperatures_anuj.feather');
actual_temp = feather.read_dataframe(dir+'Mendota_buoy_data_anuj.feather');
time_features = feather.read_dataframe(dir+'mendota_meteo.feather');

feature_mat = np.asarray(time_features.iloc[:,[1,2,3,4,5,6,7]].values.tolist());
freeze = np.asarray(glm_temp.iloc[:,-1].values.tolist());

#extracting the date features
all_dates = time_features.iloc[:,0].values.tolist();
doy = np.zeros(feature_mat.shape[0]);

#computing the day of year (doy) features from the date features
for i in range(feature_mat.shape[0]):
    doy[i] = int(all_dates[i].strftime('%j'));

#concatenating day of year features with the feature set
feature_mat = np.column_stack((feature_mat, doy, freeze));

#setting the lookback of autoencoder to be 7 days
lookback = 7;
n_features = feature_mat.shape[1];

#reshaping matrix for LSTM encoder input
mat = np.zeros((feature_mat.shape[0] - lookback, lookback, n_features));
for i in range(feature_mat.shape[0] - lookback):
    mat[i,:,:]=feature_mat[ i : i + lookback, : ];

n_obs = mat.shape[0];

#training on entire unlabelled data
train_X = mat[:n_obs, : , :]


#normalizing the training set
mean = train_X.mean(axis=0)
std = train_X.std(axis=0)
train_X = (train_X - mean) / std;
train_Y = train_X[:,-1,:];

#setting the embedding dimension to be 5.
encoding_dim = 5;

#Defining the LSTM based auto-encoder
input1 = Input(shape = (lookback, n_features))
# "encoded" is the encoded representation of the input
encoded = LSTM(encoding_dim, return_sequences = False)(input1);
repeat = RepeatVector(lookback)(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = LSTM(n_features, return_sequences = False)(repeat);

# this model maps an input to its reconstruction
autoencoder = Model(input1, decoded);

# this model maps an input to its encoded representation
encoder = Model(input1, encoded);

# create a placeholder for an encoded (10-dimensional) input
encoded_input = Input(shape = (encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-2]
decoder_lstm= autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_lstm(decoder_layer(encoded_input)))


#compiling the autoencoder and training it.
autoencoder.compile(optimizer = 'adadelta', loss = mean_squared_error);
history = autoencoder.fit(train_X , train_Y, epochs = 5000, batch_size = 256);

#generating the encoder and decoder outputs.
encoded_out = encoder.predict(train_X);
decoder_out = decoder.predict(encoded_out);
err = np.mean(np.square(train_Y - decoder_out));
embedding_features = encoded_out;

#plotting the training loss
pyplot.plot(history.history['loss'], label = 'train_loss')
pyplot.show();

spio.savemat('../Datasets/temporal_feature_autoencoder_lookback_7.mat',{'embedding_features':embedding_features});


