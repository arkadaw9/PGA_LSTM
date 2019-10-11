#tutorial on autoencoders
#https://blog.keras.io/building-autoencoders-in-keras.html

import feather
import numpy as np
from keras.layers import Input, Dense, LSTM, RepeatVector
from keras.models import Model
from keras import optimizers
from keras.losses import mean_squared_error
import scipy.io as spio


dir='';
glm_temp=feather.read_dataframe('mendota_GLM_uncal_temperatures_anuj.feather');
actual_temp=feather.read_dataframe('Mendota_buoy_data_anuj.feather');
time_features=feather.read_dataframe('mendota_meteo.feather');

feature_mat=np.asarray(time_features.iloc[:,[1,2,3,4,5,6,7]].values.tolist());
freeze=np.asarray(glm_temp.iloc[:,-1].values.tolist());
all_dates=time_features.iloc[:,0].values.tolist();
doy=np.zeros(feature_mat.shape[0]);
for i in range(feature_mat.shape[0]):
    doy[i]=int(all_dates[i].strftime('%j'));

feature_mat = np.column_stack((feature_mat, doy, freeze));

lookback=7;
n_features=feature_mat.shape[1];

mat=np.zeros((feature_mat.shape[0]-lookback,lookback,n_features));
for i in range(feature_mat.shape[0]-lookback):
    mat[i,:,:]=feature_mat[i:i+lookback,:];

n_obs=mat.shape[0];
train_X = mat[:n_obs, : , :]
#test_X= mat[n_obs:, :, :]

mean = train_X.mean(axis=0)
std = train_X.std(axis=0)
train_X = (train_X - mean) / std;
#test_X = (test_X - mean) / std;
train_Y=train_X[:,-1,:];

encoding_dim=5;
input1=Input(shape=(lookback, n_features))
# "encoded" is the encoded representation of the input
encoded=LSTM(encoding_dim, return_sequences=False)(input1);
repeat=RepeatVector(lookback)(encoded)
# "decoded" is the lossy reconstruction of the input
decoded=LSTM(n_features, return_sequences=False)(repeat);

# this model maps an input to its reconstruction
autoencoder=Model(input1,decoded);

# this model maps an input to its encoded representation
encoder=Model(input1,encoded);

# create a placeholder for an encoded (10-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-2]
decoder_lstm= autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_lstm(decoder_layer(encoded_input)))

autoencoder.compile(optimizer='adadelta',loss=mean_squared_error);
autoencoder.fit(train_X,train_Y, epochs=5000, batch_size=256);

encoded_out=encoder.predict(train_X);
decoder_out=decoder.predict(encoded_out);
#decoder_out=(decoder_out+mean)*std;
#train_Y=(train_Y+mean)*std;
err=np.mean(np.square(train_Y-decoder_out));
embedding_features=encoded_out;
#spio.savemat('temporal_feature_autoencoder_lookback_7.mat',{'embedding_features':embedding_features});



#encoded_out=encoder.predict(test_X);
#decoder_out=decoder.predict(encoded_out);
#
#
#
#embedding_features=encoded_out[:,-1,:];
#err=decoder_out-test_X
