# Physics Guided Architecture (PGA) of Neural Networks for Quantifying Uncertainty in Lake Temperature Modelling

This repository provides code for the Physics Guided Architecture (PGA) - LSTM. 

### Acknowledgements :

The implementation of the monotonicity preserving LSTM was heavily borrowed from the Keras implementation of LSTM. My sincere thanks to them!!

## Introduction :

This paper presents innovations in an emerging field of theory-guided data science where, instead of using black-box architectures, we principally embed well-known physical principles in the neural network design. We refer to this paradigm as physics-guided architecture (PGA) of neural networks. Specifically, this paper offers two key innovations in the PGA paradigm for the illustrative problem of lake temperature modeling.  First, we introduce novel *physics-informed connections* among neurons in the network to capture physics-based relationships of lake temperature. Second, we associate physical meaning to some of the neurons in the network by computing physical intermediate variables $Z$ in the neural pathway from inputs to outputs. By hard-wiring physics in the model architecture, the PGA paradigm ensures physical consistency of results regardless of small perturbations in the network weights, e.g, due to MC dropout.

For more information, please check out the paper.
## Datasets

Lake Mendota : Read, J.S., Jia, X., Willard, J., Appling, A.P., Zwart, J.A., Oliver, S.K., Karpatne, A., Hansen, G.J.A., Hanson, P.C., Watkins, W., Steinbach, M., Kumar, V. 2019, Data release: Process-guided deep learning predictions of lake water temperature: U.S. Geological Survey, http://dx.doi.org/10.5066/P9AQPIVD.

Falling Creek Reservoir (FCR) : Carey C. C., R. P. McClure, A. B. Gerling, J. P. Doubek, S. Chen, M. E. Lofton, K. D. Hamre. 2019. Time series of high-frequency profiles of depth, temperature, dissolved oxygen, conductivity, specific conductivity, chlorophyll a, turbidity, pH, oxidation-reduction potential, photosynthetic active radiation, and descent rate for Beaverdam Reservoir, Carvins Cove Reservoir, Falling Creek Reservoir, Gatewood Reservoir, and Spring Hollow Reservoir in Southwestern Virginia, USA 2013-2018. Environmental Data Initiative, https://doi.org/10.6073/pasta/8f19c5d19d816857e55077ba20570265.

If you're using these datasets in your paper, please cite the above references.

## Using the code

The repository contains code and datasets needed for training and testing the PGA-LSTM described in the paper.  Please note that we have used data from two lakes to train and test our models. The two lakes are Lake Mendota, Minnesota, USA and the Falling Creek Reservoir, Virginia, USA. Each of the code in the repository have a header named 'Mendota' or 'FCR' which represents the code for either lakes.

### Training auto-encoders to extract temporal features

To train the auto-encoder go the script 'Data Generator/[lake]\_auto_encoder_temporal_embedding_extract.py' and run it. This would load the datasets and train an autoencoder to extract temporal embeddings from the data and save the embeddings into directory path 'Datasets/[lake]\_temporal_feature_autoencoder_lookback_7.mat'.

### Data-preprocessing

After the temporal embeddings are extracted and stored in the '\Datasets' directory, you'll need to run the script 'Data Generator/[lake]\_generate_temporal_data.py' for generating the dataset by combining the temporal features extracted by the autoencoder and the available source datasets. This script also performs the discretization of the data as described in the appendix.  

This script will generate .mat file which is going to be used by our proposed PGA-LSTM model.

### Monotonicity Preserving LSTM

This script for the implementation of the monotonicity preserving LSTM can be found in the directory 'Monotonic LSTM Script/'.  Please refer to the paper for detailed description of the *monotonicity preserving* LSTM. 

### Training and Testing PGA-LSTM

Run the script 'Models/[lake]\_PGA_LSTM.py' to train and test the PGA-LSTM model for either lakes.

The script has the following tunable hyper-parameters in the '__main__' function which you can play with :

1.  tr_frac_range : list containing the values of the training fractions for which you want to generate results.
2. val_frac : Float between 0 and 1. Validation fraction. 
3. patience_val : Positive Integer. Number of epochs that produced the monitored quantity with no improvement after which training will be stopped. For PGA-LSTM, the monitored quantity is validation mean squared error.
4. num_epochs : Positive Integer. Maximum number of epochs for training
5. batch_size : Integer. Number of samples per gradient update.
6. lstm_nodes : Positive Integer. Number of LSTM units in the monotonicity preserving LSTM
7. feed_forward nodes : Positive integer. The number of hidden units in the dense layer of the monotonicity preserving LSTM.
8. mask_value : Integer.  This is the value of the mask for the date / time where ground truth is not available. Set to 0 by default. The value of the mask does not affect the results.
9. drop_frac : Float between 0 and 1. Percent of parameters being dropped out from each layer where Dropout was applied.
10. n_nodes : Positive integer. Number of units in the hidden layers which converts density predictions into temperature predictions.
11. use_GLM : Boolean {0,1}. use_GLM=1 represents that the predictions of GLM were used as additional feature.
12. use_temporal_features : Boolean {0,1}. use_temporal_feature=1 represents that the temporal features extracted by the autoencoder are used by PGA-LSTM.
13. lamda_reg : Positve. Value of lamda for L2 norm.
14. lamda_main, lamda_aux : Positive. Represents the $\lamda_{Y}$ and $\lamda_{Z}$.
15.  tol : Set to $10^-5$ by default. The tolerance value represents the threshold by which we allow the density constraints to be physically inconsistent.
16. pad_steps : size of padding layer. 



 
