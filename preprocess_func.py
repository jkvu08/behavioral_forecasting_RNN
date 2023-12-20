# -*- coding: utf-8 -*-
"""
@author: Jannet Vu

Functions for data preprocessing for the behavioral prediction dataset
1) Transform time covariates into cyclic covariates 
2) Untransform covariates back to time covariate 

Input: Cleaned, unprocessed data
Output: Processed data

"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def single_cyclic_convert(x,xmax):
    '''
    Turn value into sine and cosine components to take into account cyclic behavior

    Parameters
    ---------
    x : value to be converted
    xmax : maximum possible value x can take on 
    Returns
    -------
    xsin : sine component
    xcos : cosine component

    '''
    trans = 2*np.pi*x/xmax # scale and transform value into radians
    xsin = np.sin(trans) # calculate sine component
    xcos = np.cos(trans) # calculate cosince component
    return xsin, xcos

def single_cyclic_backtransform(xsin,xcos,xmax):
    '''
    backtransform cyclic value

    Parameters
    ----------
    xsin : sine component
    xcos : cosine component
    xmax : maximum possible value x can take on 

    Returns
    -------
    rounded backtransformed value

    '''
    isin = np.arcsin(xsin) # calculate arcsine 
    # if the cosice component is less than 0 
    if xcos < 0:
        backtrans = ((isin-np.pi)*-xmax)/(2*np.pi) # backtransform accordingly
    # if the cosine component is greater than 0 and the sine component is less than 0
    elif xsin < 0 and xcos > 0:  
        backtrans = ((isin+2*np.pi)*xmax)/(2*np.pi) # backtransform accordingly
    else: # otherwise
        backtrans = (isin*xmax)/(2*np.pi) # backtransform accordingly
    return round(backtrans,0) # round backtransformed value

def cyclic_conversion(x, xmax):
  '''
  Turn variable into sine and cosine components to take into account cyclic behavior
  Parameters
  ----------
  x: values to be converted
  xmax: maximum possible value x can take on 
  Returns
  -------
    scaler: scaler used to transform data
	xtab: 2D numpy array of X as sine and cosine components
  '''
  xtrans = 2*np.pi*x/xmax # scale and transform value into radians
  xsin = np.sin(xtrans) # get sine component
  xcos = np.cos(xtrans) # get cosine component
  xtab = np.column_stack((xsin, xcos)) # merge into 2D array
  scaler = MinMaxScaler() # generate min-max scaler to fit data between 0-1
  xtab = scaler.fit_transform(xtab) # scale the data
  return scaler, xtab

def uncyclic_conversion(xtab, xmax, scaler):
    '''
    Backtransform variable from sine and cosine components

    Parameters
    ----------
    xtab :  2D numpy array of X as sine and cosine components
    xmax : maximum possible value raw x can take on 
    scaler : scaler used to transform data

    Returns
    -------
    array of backtranformed variable

    '''
    x = scaler.inverse_transform(xtab) # unnormalize data 
    back_values = []
    for i in range(x.shape[0]):
        bvalue = single_cyclic_backtransform(x[i,0],x[i,1],xmax)
        back_values.append(bvalue)
    return np.array(back_values)
   
