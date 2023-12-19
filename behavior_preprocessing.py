"""
@author: Jannet Vu

Data preprocessing for the behavioral prediction dataset
1) Visualized raw data
2) One-hot encoded the responses
3) Transformed time covariates into cyclic covariates 
4) Normalized all covariates

Input: Cleaned, unprocessed data
Output: Processed data

"""
# load packages
import os
import pandas as pd
from pandas import read_csv
import numpy as np
from numpy import argmax
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import os
import multiprocessing as mp
import seaborn as sns
import itertools
from matplotlib import pyplot

# set working directory
os.chdir("C:\\Users\\Jannet\\Documents\\Dissertation\\codes\\behavioral_forecasting_RNN")
# set directory where results will be saved
path = "C:\\Users\\Jannet\\Documents\\Dissertation\\codes\\behavioral_forecasting_RNN\\outputs\\"

# import data
dataset = read_csv('vv_lstm_data_2021_07_12.csv', header=0, index_col = 0)

dataset.info()
# Data columns (total 25 columns):
#  #   Column                 Dtype         Description
# ---  ------                 ----          -----  
#  0   ID                     int64         record ID  
#  1   TID                    int64         group track ID   
#  2   focal                  object        focal ID    
#  3   behavior               object        behavior (R-resting, F-feeding, S-socializing, T-traveling, UNK-Unknown)
#  4   track_length           int64         no. of records in track   
#  5   track_position         int64         track timestep of record  
#  6   since_rest             float64       no. of timesteps since resting 
#  7   since_feed             float64       no. of timesteps since feeding
#  8   since_travel           float64       no. of timesteps since traveling
#  9   since_social           float64       no. of timesteps since socializing 
#  10  individual_continuity  int64         0/1 if same focal followed 
#  11  sex                    float64       sex  
#  12  adults                 float64       no. of adults in group
#  13  infants                float64       no. of infants in group
#  14  juveniles              float64       no. of juveniles in group
#  15  reproductive_state     object        reproductive phase (GES-gestatation, LAC- lactation, NR- nonbreeding, MEG- mating/early gestation)
#  16  rain                   float64       hourly rainfall (mm)
#  17  temperature            float64       hourly temp (C)
#  18  flower_count           float64       biweekly no. of trees flowering in phenology plots 
#  19  fruit_count            float64       biweekly no. of trees fruiting in phenology plots 
#  20  flower_shannon         float64       biweekly diversity of flowering trees in phenology plots 
#  21  fruit_shannon          float64       biweekly diversity of fruiting trees in phenology plots 
#  22  minutes                int64         minute of the day
#  23  doy                    int64         doy of the year
#  24  year                   int64         year

# colums 0-2 are identifiers
# columns 3 is the response 
# column 4-24 are the predictors

# visualize time series of continuous predictors
# select predictors to plot
groups = list(itertools.chain(range(16,22), range(12,15),range(6,9),range(4,6)))

# plot subsample of predictor timeseries
fig, axs = pyplot.subplots(14,1,figsize=(8,5), sharex = True, sharey = False, squeeze = True, dpi = 300)
i = 1 # create counter
for group in groups: # for each group
    pyplot.subplot(len(groups), 1, i) # divide the plot space 
    # 12951:18173 selected as subset of timeseries to plot, can change to plot different subsection of time series 
    g = sns.lineplot(dataset.loc[12951:18173, 'doy'],dataset.iloc[12951:18173, group], ci = None, color = 'gray') # plot the predictors
    g.tick_params(axis = 'both',labelsize = 7)
    g.set_ylabel(dataset.columns[group], fontsize = 7, rotation = 40, ha='right')
    g.set_xlabel('doy', fontsize = 7)
    i += 1 # update counter
pyplot.show()

# output figure
fig.savefig(path+'data_timeseries_samples.png', dpi=300)

## one hot encode multiclass categorical data
# function to decode a one hot encoded string
def one_hot_decode(encoded_seq):
  """
  Reverse one_hot encoding
    
  Parameters
  ----------
  encoded_seq: array of one-hot encoded data 
	Returns
  -------
	series of labels
	"""
  return [argmax(vector) for vector in encoded_seq] # returns the index with the max value

def cyclic_conversion(x, xmax):
  '''
  Turn variable into sine and cosine components to take into account cyclic behavior
  Parameters
  ----------
  x: values to be converted
  xmax: maximum possible value x can take on 
  Returns
  -------
	xtab: 2D numpy array of X as sine and cosine components
	'''
  xsin = np.sin(2*np.pi*x/xmax) # get sine component
  xcos = np.cos(2*np.pi*x/xmax) # get cosine component
  xtab = np.column_stack((xsin, xcos)) # merge into 2D array
  scaler = MinMaxScaler() # generate min max scaler to fit data betewen 0-1
  xtab = scaler.fit_transform(xtab) # scale the data
  return scaler, xtab
    

# convert string to numeric using factorization
bcode = pd.factorize(dataset.behavior.values,na_sentinel=None, sort=True)
rcode = pd.factorize(dataset.reproductive_state.values,na_sentinel=None, sort = True)

# one-hot encode the data 
b_encoded = to_categorical(bcode[0], 5)
b = pd.DataFrame(b_encoded)
b.loc[(b[4] == 1)] = -1 # replace unknowns with missing values
b_encoded = b.loc[:,range(0,4)] # get rid of the unknown column
del b

# encode the reproductive state
r_encoded = to_categorical(rcode[0], 4)

# get the code-numeric association
bcode = bcode[1]
rcode = rcode[1]

# merge onehot coded data into an array
data = np.column_stack((dataset.individual_continuity, b_encoded, r_encoded, dataset.sex, dataset.fragment))

## transform quantitative data by scaling to mean center and normalizing
# subset out data to be scaled
scale_data = dataset.loc[:,['track_length', 'track_position','since_rest','since_feed','since_travel',
                           'since_social','adults', 'infants','juveniles','rain','temperature','flower_count',
                           'fruit_count', 'flower_shannon','fruit_shannon','year']] 

scaler_cont = MinMaxScaler()
scale_data = scaler.fit_transform(scale_data)
data = np.column_stack((data,scale_data))

# format cyclic data using sinusoidal encoding
min_scale, min_x = cyclic_conversion(dataset.loc[:,'minutes'], 1440)
doy_scale, doy_x = cyclic_conversion(dataset.loc[:,'minutes'], 365)

# add to feature dataset
data = np.colum_stack((data, min_x, doy_x))

# replace NA values, use a number that is outside of the scaled range
data = np.nan_to_num(data,copy = False, nan = -1) # used -1 in this case

# turn into dataframe
df = pd.DataFrame(data)
# add column names
df.columns = ['individual_continuity','feed','rest','social','travel',
                 'gestation','lactation','mating','nonreproductive', 'sex','length', 
                 'position','since_rest','since_feed','since_travel','since_social',
                 'adults', 'infants','juveniles','rain','temperature','flower_count',
                 'fruit_count', 'flower_shannon','fruit_shannon','years',
                 'minutes_sin','minutes_cos','doy_sin','doy_cos','fragment']

# add identifiers at at the beginning of the standardized dataframe
df = pd.concat([dataset[['ID','TID','track_position','track_length','focal', 'year']].reset_index(drop=True), df.reset_index(drop=True)], axis = 1)

df.to_csv('behavior_formatted.csv')
