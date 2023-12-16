# Multiclass behavioral classification
Multi-class behavioral predictions using recurrent neural networks.

# Files
behavior-preprocessing.py: Data preprocessing steps 
single_vanilla_rnn.py: Behavioral prediction using vanilla RNN model 
single_ende_rnn.py: Behavioral prediction using encoder-decoder RNN model analyses
hyperopt_vanilla_rnn.py: Hyperparameter optimization for the vanilla RNN behavioral forecasting model
hyperopt_ende_rnn.py: Hyperparameter optimization for the encoder-decoder RNN behavioral forecasting model 
hyperopt_visual_assess.py: Wrapper to visualize the hyperparameter optimization runs
permutation_feat_importance.py: Permutation feature importance analysis
null_models.py: Generate null model predictions and assess performance. Two null models were used in this study: 
 - Null0 makes predictions by drawing from the behavior frequency distributions
 - Null1 makes predictions by drawing based on the transition likelihoods between behaviors

# Requirements
Tensorflow

# Data
Data used for study is not publically available, edit