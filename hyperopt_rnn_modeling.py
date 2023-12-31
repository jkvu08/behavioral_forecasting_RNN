

###########################
#### Hyperoptimization ####
###########################
def hyp_nest(params, features, targets):
    '''
    construct vanilla RNN or encoder-decode RNN based on parameter dictionary specifications

    Parameters
    ----------
    params : dict, dictionary of paramters and hyperparameters
    features : int, number of features used for prediction
    targets : int, number of targets (classes) predicted

    Raises
    ------
    Exception
        something other than 'VRNN' designating vanilla RNN or 'ENDE' designating encoder-decoder RNN was specified in params['atype']

    Returns
    -------
    model : RNN model

    '''
    if params['atype'] == 'VRNN':
        model = build_rnn(features, 
                          targets, 
                          lookback = params['lookback'], 
                          neurons_n=params['neurons_n'],
                          hidden_n=[params['hidden_n0'], params['hidden_n1']],
                          lr_rate =params['learning_rate'],
                          d_rate = params['dropout_rate'],
                          layers = params['hidden_layers'], 
                          mtype = params['mtype'], 
                          cat_loss = params['loss'])
    elif params['atype'] == 'ENDE':
        model = build_ende(features, 
                           targets, 
                           lookback = params['lookback'], 
                           n_outputs = params['n_outputs'], 
                           neurons_n=params['neurons_n'],
                           hidden_n=[params['hidden_n0'], params['hidden_n1']],
                           td_neurons = params['td_neurons'], 
                           lr_rate =params['learning_rate'],
                           d_rate = params['dropout_rate'],
                           layers = params['hidden_layers'], 
                           mtype = params['mtype'],
                           cat_loss = params['loss'])
    else:
        raise Exception ('invalid model architecture')    
    return model

def hyperoptimizer_rnn(params):
    """
    hyperparameter optimizer objective function to be used with hyperopt

    Parameters
    ----------
    params : hyperparameter search space

    Returns
    -------
    dict
        loss: loss value to optimize through minimization (i.e., validation loss)
        status: default value for hyperopt
        params: the hyperparameter values being tested
        val_loss: validation loss
        train_loss: training loss
        train_f1: training f1

    """
    targets=4 # set number of targets (4 behavior classes)
    train, test = split_dataset(datasub, 2015) # split the data by year
    # format training data
    train_X, train_y, train_dft = to_supervised(data = train.iloc[:,7:33], 
                                                TID = train['TID'], 
                                                window = 1, 
                                                lookback = params['lookback'], 
                                                n_output=params['n_output']) 
    # format testing data
    test_X, test_y, test_dft = to_supervised(data = test.iloc[:,7:33], 
                                             TID = test['TID'],
                                             window = 1, 
                                             lookback = params['lookback'], 
                                             n_output = params['n_outputs'])
    
    # if encoder-decode model and predict 1 timestep, reconfigure 2d y to 3d
    if params['atype'] == 'ENDE' & params['n_outputs'] == 1:
        test_y = test_y[:,newaxis,:]
        train_y = train_y[:,newaxis,:]
    
    # assign and format feature set
    if params['predictor'] == 'full': # use full set of features
        features=26 # set feature number
    elif params['predictor'] == 'behavior': # use only prior behaviors as features
        features=4 # set features
        # subset only prior behavior features
        train_X = np.copy(train_X[:,:,0:4]) 
        test_X = np.copy(test_X[:,:,0:4])
    else: # use the extrinsic conditions
        features = 17
        # subset only extrinsic features
        train_X = np.copy(train_X[:,:,np.r_[4:9,11:18,21:26]])
        test_X = np.copy(test_X[:,:,np.r_[4:9,11:18,21:26]])    
    model = hyp_nest(params, features, targets) # build model
    # fit model and extract evaluation epochs, loss and metrics
    _, avg_eval = eval_iter(model, 
                            params, 
                            train_X, 
                            train_y, 
                            test_X, 
                            test_y, 
                            patience = params['patience'], 
                            max_epochs = params['max_epochs'], 
                            atype = params['atype'], 
                            n = params['iters'])   
    # convert evaluation loss and metrics to dictionary
    obj_dict = avg_eval[1:].to_dict() # don't need average epochs run (first entry)
    obj_dict['params'] = params # add parameters to dictionary
    obj_dict['status'] = STATUS_OK # required for objective function 
    print('Best validation for trial:', obj_dict['val_f1']) # print the validation score
    return obj_dict


def hyperoptimizer_vrnn(params):
    """
    hyperparameter optimizer objective function to be used with hyperopt

    Parameters
    ----------
    params : hyperparameter search space

    Returns
    -------
    dict
        loss: loss value to optimize through minimization (i.e., validation loss)
        status: default value for hyperopt
        params: the hyperparameter values being tested
        val_loss: validation loss
        train_loss: training loss
        train_f1: training f1

    """
    targets=4 # set number of targets (4 behavior classes)
    train, test = split_dataset(datasub, 2015) # split the data by year
    # format training data
    train_X, train_y, train_dft = to_supervised(data = train.iloc[:,7:33], 
                                                TID = train['TID'], 
                                                window = 1, 
                                                lookback = params['lookback'], 
                                                n_output=params['n_output']) 
    # format testing data
    test_X, test_y, test_dft = to_supervised(data = test.iloc[:,7:33], 
                                             TID = test['TID'],
                                             window = 1, 
                                             lookback = params['lookback'], 
                                             n_output = params['n_output'])
    # assign and format feature set
    if params['predictor'] == 'full': # use full set of features
        features=26 # set feature number
    elif params['predictor'] == 'behavior': # use only prior behaviors as features
        features=4 # set features
        # subset only prior behavior features
        train_X = np.copy(train_X[:,:,0:4]) 
        test_X = np.copy(test_X[:,:,0:4])
    else: # use the extrinsic conditions
        features = 17
        # subset only extrinsic features
        train_X = np.copy(train_X[:,:,np.r_[4:9,11:18,21:26]])
        test_X = np.copy(test_X[:,:,np.r_[4:9,11:18,21:26]])    
    model = hyp_rnn_nest(params, features, targets) # build model
    # fit model and extract monitoring metrics
    _, val_f1,val_loss,train_f1, train_loss = eval_f1_iter(model, 
                                                           params, 
                                                           train_X, 
                                                           train_y, 
                                                           test_X, 
                                                           test_y, 
                                                           patience = 30, 
                                                           atype ='VRNN', 
                                                           max_epochs = 200, 
                                                           n=1) 
    print('Best validation for trial:', val_f1) # print the validation score
    return {'loss': -val_f1,
            'status': STATUS_OK,  
            'params': params,
            'val_loss': val_loss,
            'val_f1': val_f1,
            'train_loss': train_loss,
            'train_f1':train_f1}   

space_vrnn = {'covariate'              : 'full',
              'drate'                  : hp.quniform('drate',0.1,0.9,0.1),
              'neurons_n'              : scope.int(hp.quniform('neurons_n',5,50,5)),
              'n_output'               : 1,
              'learning_rate'          : 0.001,
              'hidden_layers'          : scope.int(hp.choice('layers',[0,1])),
              'hidden_n0'              : scope.int(hp.quniform('hidden_n0',5,50,5)),
              'lookback'               : scope.int(hp.quniform('lookback',1,23,1)),
              'epochs'                 : 200,
              'batch_size'             : 512,
              'weights_0'              : hp.quniform('weights_0',1,5,0.5),
              'weights_1'              : 1,
              'weights_2'              : scope.int(hp.quniform('weights_2',1,25,1)),
              'weights_3'              : scope.int(hp.quniform('weights_3',1,10,1)),
              'mtype'                  : 'LSTM'
              }

space_vrnn = {'covariate'              : 'full',
              'drate'                  : hp.quniform('drate',0.1,0.5,0.1),
              'neurons_n'              : scope.int(hp.quniform('neurons_n',5,50,5)),
              'n_output'               : 1,
              'learning_rate'          : 0.001,
              'hidden_layers'          : 0,
              'hidden_n0'              : 0,
              'lookback'               : scope.int(hp.quniform('lookback',1,23,1)),
              'epochs'                 : 200,
              'batch_size'             : 512,
              'weights_0'              : hp.quniform('weights_0',1,3,0.5),
              'weights_1'              : 1,
              'weights_2'              : hp.quniform('weights_2',1,12,1),
              'weights_3'              : hp.quniform('weights_3',1,5,0.5),
              'mtype'                  : 'GRU'
              }

def hyperoptimizer_ende(params):
    """
    hyperparameter optimizer objective function to be used with hyperopt

    Parameters
    ----------
    params : hyperparameter search space
    
    Returns
    -------
    dict
        loss: loss value to optimize through minimization (i.e., -validation f1)
        status: default value for hyperopt
        params: the hyperparameter values being tested
        val_loss: validation loss
        train_loss: training loss
        train_f1: training f1

    """
    targets=4 # set targets
    train, test = split_dataset(datasub, 2015) # split the data
    train_X, train_y, train_dft = to_supervised(data = train.iloc[:,7:33], TID = train['TID'], window = 1, lookback = params['lookback'], n_output=params['n_output']) # format training data
    test_X, test_y, test_dft = to_supervised(data = test.iloc[:,7:33], TID = test['TID'],window = params['n_output'], lookback = params['lookback'], n_output = params['n_output']) # format testing data
    # format target as 3D array if it isn't already
    if params['n_output'] == 1:
        test_y = test_y[:,newaxis,:]
        train_y = train_y[:,newaxis,:]
    
    # extract appropriate covariates
    if params['covariate'] == 'full':
        features=26 # set features
    elif params['covariate'] == 'behavior':    
        features=4 # set features
        train_X = np.copy(train_X[:,:,0:4])
        test_X = np.copy(test_X[:,:,0:4])
    else:
        features = 17
        train_X = np.copy(train_X[:,:,np.r_[4:9,11:18,21:26]])
        test_X = np.copy(test_X[:,:,np.r_[4:9,11:18,21:26]])   
       
    model = hyp_ende_nest(params, features, targets) # build model based on hyperparameters
    
    _, val_f1,val_loss,train_f1, train_loss = eval_f1_iter(model, params, train_X, train_y, test_X, test_y, patience = 30, atype ='ENDE', max_epochs = 200, n=1) # fit model and extract monitoring metrics
    print('Best validation for trial:', val_f1) # print the validation score
    return {'loss': val_loss,
            'status': STATUS_OK,  
            'params': params,
            'val_loss': val_loss,
            'val_f1': val_f1,
            'train_loss': train_loss,
            'train_f1':train_f1}    

def run_trials(filename, objective, space, rstate, initial = 20, trials_step = 1):
    """
    Run and save trials indefinitely until manually stopped. 
    Used to run trials in small batches and periodically save to file.
    
    Parameters
    ----------
    filename : trial filename
    objective : objective
    space : parameters
    rstate: set random state for consistency across trials
    initial: initial number of trials, should be > 20 
    trials_steps: how many additional trials to do after loading saved trials.
    
    Returns
    -------
    None.

    """
    max_trials = initial  # set the initial trials to run (should be at least 20, since hyperopt selects parameters randomly for the first 20 trials)
    try:  # try to load an already saved trials object, and increase the max
        trials = joblib.load(filename) # load file 
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step # increase the max_evals value
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # if trial file cannot be found
        trials = Trials() # create a new trials object
    # run the search
    best = fmin(fn=objective, # objective function to minimize
                space=space, # parameter space to search 
                algo=tpe.suggest, # search algorithm, use sequential search
                max_evals=max_trials, # number of maximum trials
                trials=trials, # previously run trials
                rstate = np.random.default_rng(rstate)) # seed
    print("Best:", best)
    print("max_evals:", max_trials)
    joblib.dump(trials, filename) # save the trials object
    return max_trials

space_ende = {'covariate'              : 'behavior',
              'drate'                  : hp.quniform('drate',0.1,0.9,0.1),
              'neurons_n0'             : scope.int(hp.quniform('neurons_n0',5,50,5)),
              'neurons_n1'             : scope.int(hp.quniform('neurons_n1',5,50,5)),
              'n_output'               : 1,
              'learning_rate'          : 0.001,
              'td_neurons'             : scope.int(hp.quniform('td_neurons',5,50,5)),
              'hidden_layers'          : scope.int(hp.choice('layers',[0,1])),
              'hidden_n0'              : scope.int(hp.quniform('hidden_n0',5,50,5)),
              'lookback'               : scope.int(hp.quniform('lookback',1,23,1)),
              'epochs'                 : 200,
              'batch_size'             : 512,
              'weights_0'              : hp.quniform('weights_0',1,5,0.5),
              'weights_1'              : 1,
              'weights_2'              : scope.int(hp.quniform('weights_2',1,25,1)),
              'weights_3'              : scope.int(hp.quniform('weights_3',1,10,1)),
              'mtype'                  : 'GRU'
              }

# params = {'covariate': 'full',
#           'drate': 0.3,
#           'neurons_n0': 5,
#           'neurons_n1': 0,
#           'neurons_n': 5,
#           'n_output': 1,
#           'learning_rate': 0.001,
#           'hidden_layers': 0,
#           'hidden_n0': 10,
#           'hidden_n': 50,
#           'td_neurons': 5,
#           'lookback': 21,
#           'epochs': 200,
#           'batch_size': 512,
#           'weights_0': 1,
#           'weights_1': 1,
#           'weights_2': 3,
#           'weights_3': 1,
#           'mtype': 'GRU'}

# train, test = split_dataset(datasub, 2015) # split the data
# train_X, train_y, train_dft = to_supervised(data = train.iloc[:,7:33], TID = train['TID'], window = 1, lookback = params['lookback'], n_output=params['n_output']) # format training data
# test_X, test_y, test_dft = to_supervised(data = test.iloc[:,7:33], TID = test['TID'],window = params['n_output'], lookback = params['lookback'], n_output = params['n_output']) # format testing data


# model = hyp_ende_nest(params,26,4)
# #weights = dict(zip([0,1,2,3], [params['weights_0'], params['weights_1'], params['weights_2'], params['weights_3']]))
       
# start_time = time.perf_counter()
# history3 = model.fit(train_X, train_y, 
#                             epochs = 200, 
#                             batch_size = params['batch_size'],
#                             verbose = 2,
#                             shuffle=False,
#                             validation_data = (test_X, test_y),
#                             sample_weight = sample_weights)
#                             #class_weight = weights)
#                           #  callbacks = EarlyStopping(patience= 30, monitor='val_loss', mode = 'min', restore_best_weights=True, verbose=0))
# print((time.perf_counter()-start_time)/60)

# plot_fun(history3)

# def plot_fun(history):
#     xlength = len(history.history['val_loss'])
#     fig, ax = pyplot.subplots(4,2,sharex = True, sharey = False, figsize = (8,8))
#     pyplot.subplot(4,2,1)
#     pyplot.plot(range(xlength), history.history['loss'],label ='train')
#     pyplot.plot(range(xlength), history.history['val_loss'], label ='valid')
#     pyplot.legend(['train', 'valid'])
#     pyplot.title('loss')
#     pyplot.subplot(4,2,2)
#     pyplot.plot(range(xlength), history.history['f1'], label ='train')
#     pyplot.plot(range(xlength), history.history['val_f1'], label ='valid')
#     pyplot.title('f1 score')
#  #   pyplot.subplot(4,2,3)
#   #  pyplot.plot(range(xlength), history.history['categorical_accuracy'], label ='train')
#    # pyplot.plot(range(xlength), history.history['val_categorical_accuracy'], label ='valid')
#     #pyplot.title('categorical accuracy')
#     pyplot.subplot(4,2,4)
#     pyplot.plot(range(xlength), history.history['Accuracy'], label ='train')
#     pyplot.plot(range(xlength), history.history['val_Accuracy'], label ='valid')
#     pyplot.title('accuracy')
#     pyplot.subplot(4,2,5)
#     pyplot.plot(range(xlength), history.history['precision'], label ='train')
#     pyplot.plot(range(xlength), history.history['val_precision'], label ='valid')
#     pyplot.title('precision')
#     pyplot.subplot(4,2,6)
#     pyplot.plot(range(xlength), history.history['recall'], label ='train')
#     pyplot.plot(range(xlength), history.history['val_recall'], label ='valid')
#     pyplot.title('recall')
#     pyplot.subplot(4,2,7)
#     pyplot.plot(range(xlength), history.history['ROC'], label ='train')
#     pyplot.plot(range(xlength), history.history['val_ROC'], label ='valid')
#     pyplot.title('ROC')
#     pyplot.subplot(4,2,8)
#     pyplot.plot(range(xlength), history.history['PR'], label ='train')
#     pyplot.plot(range(xlength), history.history['val_PR'], label ='valid')
#     pyplot.title('PR')
#     fig.tight_layout()

def iter_trials_vrnn(seed):
    g = 0
    while g < 1000:
        g = run_trials(filename = 'vrnn' + '_' +space_vrnn['mtype'] +'_' + space_vrnn['covariate'] + '_'+str(seed)+'.pkl',objective =hyperoptimizer_vrnn, space =space_vrnn, rstate =seed, initial=25, trials_step=3)
    
def iter_trials_ende(seed):
    g = 0
    while g < 1000:
        g = run_trials(filename = 'ende' + '_' +space_ende['mtype'] +'_' + space_ende['covariate'] + '_'+str(seed)+'.pkl',objective =hyperoptimizer_ende, space =space_ende, rstate =seed, initial=25, trials_step=3)

# ray.init(ignore_reinit_error=True, logging_level=logging.ERROR)
# @ray.remote
# class Simulator(object):
#     def __init__(self,seed):
#         import tensorflow as tf
#         import keras
#         from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, GRU, Conv1D, Activation, RepeatVector, TimeDistributed, Flatten, MaxPooling1D, ConvLSTM2D
#         #from tensorflow.keras.preprocessing import sequence
#         from tensorflow.keras.optimizers import Adam
#         from tensorflow.keras.models import Sequential, Model
#         from tensorflow_addons.metrics import F1Score
#         #from tensorflow.keras.utils import to_categorical
#         # import CategoricalAccuracy, CategoricalCrossentropy
#         #from tensorflow.compat.v1.keras.layers import mean_per_class_accuracy
#         from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
#         from tensorflow.keras.callbacks import EarlyStopping
#         from keras.callbacks import Callback
#         import keras.backend as K
#         from tensorflow.compat.v1 import ConfigProto, InteractiveSession, Session

#         # num_CPU = 1
#         # num_cores = 7
#         # config = ConfigProto(intra_op_parallelism_threads = num_cores,
#         #                       inter_op_parallelism_threads = num_cores,
#         #                       device_count={'CPU':num_CPU})
        
#         gpus = tf.config.experimental.list_physical_devices('GPU')
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
    
#         # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3,allow_growth=True) 
#         # config = ConfigProto(gpu_options)
#         # self.sess = Session(config=config)
#         self.seed = seed

#     def iter_trials_vrnn(self):
#         g = 0
#         while g < 1000:
#             g = run_trials(filename = 'vanilla_rnn_vv_trials_seed'+str(self.seed)+'.pkl',objective =hyperoptimizer_vrnn, space =space_vrnn, rstate =self.seed, initial=25, trials_step=2)
#        # self.sess.close()
        
#     def iter_trials_ende(self):
#         g = 0
#         while g < 1000:
#             g = run_trials(filename = 'ende_vv_trials_seed'+str(self.seed)+'.pkl',objective =hyperoptimizer_ende, space =space_ende, rstate =self.seed, initial=25, trials_step=3)
    
#     def iter_trials_vbonly(self):
#         g = 0
#         while g < 1000:
#             g = run_trials(filename = 'vrnn_bonly_vv_trials_seed'+str(self.seed)+'.pkl',objective =hyperoptimizer_vrnn_bonly, space =space_bonly, rstate =self.seed, initial=25, trials_step=3)

# start = time.perf_counter()
# simulators = [Simulator.remote(a) for a in [123,619,713]]
# results = ray.get([s.iter_trials_vrnn.remote() for s in simulators])
# finish = time.perf_counter()
# print('Took '+str((finish-start)/(3600)) + 'hours')


# 144,302,529
# # load up trials
# ende20 = joblib.load("ende_vv_trials_seed20.pkl")
# ende51 = joblib.load("ende_vv_trials_seed51.pkl")
# ende90 = joblib.load("ende_vv_trials_seed90.pkl")

# vrnn20 = joblib.load("vanilla_rnn_vv_trials_seed20.pkl")
# vrnn16 = joblib.load("vanilla_rnn_vv_trials_seed16.pkl")
# vrnn06 = joblib.load("vanilla_rnn_vv_trials_seed6.pkl")

# seeds = random.sample(range(100000),3)


# current = time.perf_counter()
# run_trials(filename = 'ende' + '_' +space_ende['mtype'] +'_' + space_ende['covariate'] + '_'+str(114)+'.pkl',objective =hyperoptimizer_ende, space =space_ende, rstate =114, initial=2, trials_step=2)
# #run_trials(filename = 'vrnn' + space_vrnn['mtype'] +'_' + space_vrnn['covariate'] + '_'+str(seeds[0])+'.pkl',objective =hyperoptimizer_vrnn, space =space_vrnn, rstate =seeds[0], initial=2, trials_step=2)
# print((time.perf_counter()-current)/60)

# run_trials(filename = 'vrnn' + space_vrnn['mtype'] +'_' + space_vrnn['covariate'] + '_'+str(seeds[1])+'.pkl',objective =hyperoptimizer_vrnn, space =space_vrnn, rstate =seeds[1], initial=25, trials_step=2)
# run_trials(filename = 'vrnn' + space_vrnn['mtype'] +'_' + space_vrnn['covariate'] + '_'+str(seeds[2])+'.pkl',objective =hyperoptimizer_vrnn, space =space_vrnn, rstate =seeds[2], initial=25, trials_step=2)

# run_trials(filename = 'vanilla_rnn_vv_trials_seed'+str(312)+'.pkl',objective =hyperoptimizer_vrnn, space =space_vrnn, rstate =312, initial=2, trials_step=2)
# run_trials(filename = 'vanilla_rnn_vv_trials_seed'+str(223)+'.pkl',objective =hyperoptimizer_vrnn, space =space_vrnn, rstate =223, initial=25, trials_step=2)
# run_trials(filename = 'vanilla_rnn_vv_trials_seed'+str(969)+'.pkl',objective =hyperoptimizer_vrnn, space =space_vrnn, rstate =969, initial=25, trials_step=2)

current = time.perf_counter()
iter_trials_vrnn(random.sample(range(100000),1)[0])
print((time.perf_counter()-current)/60)

current = time.perf_counter()
iter_trials_ende(random.sample(range(100000),1)[0])
print((time.perf_counter()-current)/3600)

current = time.perf_counter()
iter_trials_vrnn(42577)
print((time.perf_counter()-current)/60)

iter_trials_vrnn(523)

current = time.perf_counter()
iter_trials_ende(56924)
print((time.perf_counter()-current)/3600)


# test model construction wrapper, should generate same model architecture as prior model
# specify parameters/hyperparameters
# hidden neuron variables differ from prior parameter specification
params = {'atype': 'VRNN',
          'mtype': 'GRU',
          'lookback': lookback, # also added lookback as parameter in dictionary
          'hidden_layers': 1,
          'neurons_n': 20,
          'hidden_n0': 10,
          'hidden_n1': 10,
          'learning_rate': 0.001,
          'dropout_rate': 0.3,               
          'loss': False,
          'epochs': 5,
          'batch_size': 512,
          'weights_0': 1,
          'weights_1': 1,
          'weights_2': 3,
          'weights_3': 1}

model = bmf.hyp_nest(params, features, targets)
model.summary() # looks identical to prior model as expected
# Model: "sequential_4"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Masking (Masking)            (None, 5, 26)             0         
# _________________________________________________________________
# GRU (GRU)                    (None, 20)                2880      
# _________________________________________________________________
# dropout_8 (Dropout)          (None, 20)                0         
# _________________________________________________________________
# dense_4 (Dense)              (None, 10)                210       
# _________________________________________________________________
# dropout_9 (Dropout)          (None, 10)                0         
# _________________________________________________________________
# Output (Dense)               (None, 4)                 44        
# =================================================================
# Total params: 3,134
# Trainable params: 3,134
# Non-trainable params: 0
# _________________________________________________________________
