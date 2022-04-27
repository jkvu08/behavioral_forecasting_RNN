"""
@author: Jannet Vu

Data preprocessing for the behavioral prediction dataset
1) Visualized raw data
2) One-hot encoded the responses
3) Transformed time covariates into cyclic covariates 
4) Mormalized all covariates

Input: Cleaned, unprocessed data
Output: Processed data

"""
import pandas as pd
from pandas import read_csv
import numpy as np
from numpy import argmax
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import os
import multiprocessing as mp
import seaborn as sns

## one hot encode categorical data
# decode a one hot encoded string
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
    
# import dataframe
# data was previously clean and standardized in R
dataset = read_csv('data.csv', header=0, index_col = 0)

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
