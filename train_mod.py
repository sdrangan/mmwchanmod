# -*- coding: utf-8 -*-
"""
chan_mod.py:  VAE training of channel model data

We want to model the distribution of the channel data.
In this case, each channel is a vector x[i,:] of the npaths_max
received powers.  x[i,j] = 0 corresponds to no path.

Also u[i,:] is the condition of the link.  In this case, ç
u[i,:] = [log10(d), d] where d is the distance.

We want to find the conditional pdf p(x|u).
Note that a traditional VAE only finds p(x).


Model this density as x = decoder(z,u)  where z is N(0,I)

"""
import numpy as np
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import sklearn.preprocessing

from models import CondVAE, ChanMod


"""
Parameters
"""
nlatent = 20   # number of latent variables
fit_link = False
fit_path = True
npaths_max = 20


# Load the data
fn = 'train_test_data.p'
with open(fn, 'rb') as fp:
    train_data,test_data,pl_max = pickle.load(fp)
 

"""
Train the link classifier
"""
K.clear_session()

# Construct the channel model object
chan_mod = ChanMod(nlatent=nlatent,pl_max=pl_max, npaths_max=npaths_max,\
                   nunits_enc=(100,50), nunits_dec=(50,100),  nunits_link=(50,25),\
                   init_bias_stddev=10.0, init_kernel_stddev=10.0)



if fit_link:
    # Build the link model
    chan_mod.build_link_mod()
    
    # Fit the link model 
    chan_mod.fit_link_mod(train_data, test_data, lr=1e-3)

    # Save the link classifier model
    chan_mod.save_link_model()
    
else:
    # Load the link model
    chan_mod.load_link_model()


"""
Train the path loss model
"""
if fit_path:
    chan_mod.build_path_mod()
    chan_mod.fit_path_mod(train_data, test_data,lr=1e-3,epochs=200)
    
    # Save the path loss model
    chan_mod.save_path_model()




