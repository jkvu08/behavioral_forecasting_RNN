
# 5	Forecasting the activity of the black-and-white ruffed lemur (Varecia variegata) using recurrent neural networks

Predict the sub-daily multi-state behavior of wild black-and-white ruffed lemurs based on their internal and external conditions. 
## Acknowledgements
 - [Almeida & Azkune 2018, Predicting human behaviour with recurrent neural networks](https://www.mdpi.com/2076-3417/8/2/305)
 - [Gomes et al. 2020, An Amazon stingless bee foraging activity predicted using recurrent artificial neural networks and attribute selection](https://www.nature.com/articles/s41598-019-56352-8)
 - [@Jason Brownlee](https://machinelearningmastery.com/)
 - [@Krish Naik](https://github.com/krishnaik06)
 


## Authors

- [Jannet K. Vu](https://www.github.com/jkvu08)
- [Sheila M. Holmes](https://www.researchgate.net/profile/Sheila-Holmes)
- [Steig E. Johnson](https://www.steigjohnson.com/)
- [Edward E. Louis Jr.](https://www.researchgate.net/profile/Edward-Louis)



## Content

    1. behavior_preprocessing.py - Code for preprocessing data for behaviral prediction models 
    2. hyperopt_vanilla_rnn.py - Code for optimizing vanilla RNN parameters using Bayesian hyperparameter optimization
    3. hyperopt_ende_rnn.py - Code for optimizing encoder-decoder RNN parameters using Bayesian hyperparameter optimization
    5. null_models.py - Code for behavioral prediction using Markov model and activity distribution model
    6. permutation_feat_importance.py - Code for permutation feature importance to determine the relative contribution of each feature in each timestep on the behavioral prediction
    7. single_ende_rnn.py - Code for behavioral prediction using encoder-decoder RNN
    8. single_vanilla_rnn.py - Code for behavioral prediction using vanilla RNN
    
## Support

For support, email jannetkimvu@ucla.edu.

