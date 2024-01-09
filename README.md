
# Forecasting the activity of wild black-and-white ruffed lemurs using recurrent neural networks

Predict the multi-class behaviors (4 classes) of wild black-and-white ruffed lemurs based on their internal state (e.g., prior behavior, sex, reproductive state) and external condition (e.g., weather, group composition, resource availability), using using recurrent neural networks.
## Acknowledgements
 - [Almeida and Azkune 2018, Predicting human behaviour with recurrent neural networks](https://www.mdpi.com/2076-3417/8/2/305)
 - [Gomes et al. 2020, An Amazon stingless bee foraging activity predicted using recurrent artificial neural networks and attribute selection](https://www.nature.com/articles/s41598-019-56352-8)
- [@Michal Haltuf (Best loss function for F1-score metric)](https://www.kaggle.com/code/rejpalcz/best-loss-function-for-f1-score-metric/notebook)
- [@Krish Naik](https://www.youtube.com/@krishnaik06)
- [@Jason Brownlee (Machine Learning Mastery)](https://machinelearningmastery.com/start-here/)

 

## Authors

- [Jannet K. Vu](https://www.github.com/jkvu08)
- [Sheila M. Holmes](https://www.researchgate.net/profile/Sheila-Holmes)
- [Steig E. Johnson](https://www.steigjohnson.com/)
- [Edward E. Louis Jr.](https://www.researchgate.net/profile/Edward-Louis)



## Content

    1. behavior_model_func.py - Functions for running and evaluating RNN models  
    2. behavior_prediction_test_train_splitting.py - Split data into testing and training sets
    3. behavior_preprocessing.py - Preprocess data for behavioral prediction models
    4. hyperopt_rnn_modeling.py - Run Bayesian hyperparameter optimization to find the best performing parameters for the RNN models 
    5. hyperopt_vis_func.py - Functions for visualizing hyperparameter optimization results
    6. null_models.py - Predict lemur behaviors based on behavioral activity distribution and transition matrices
    7. preprocess_func.py - Functions for preprocessing data
    8. rnn_prediction_assess.py - Evaluate RNN performance during testing phase
    9. single_ende_rnn.py - Run encoder-decorder RNN models 
    10. single_vanilla_rnn.py - Run vanilla RNN models
    11. outputs - Folder containing results
## Support

For support, email jannetkimvu@ucla.edu.

