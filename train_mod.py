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
import argparse

from models import ChanMod

"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Trains the channel model')
parser.add_argument('--nlatent',action='store',default=20,type=int,\
    help='number of latent variables')
parser.add_argument('--npaths_max',action='store',default=20,type=int,\
    help='max number of paths per link')
parser.add_argument('--nepochs_link',action='store',default=50,type=int,\
    help='number of epochs for training the link model')
parser.add_argument('--lr_link',action='store',default=1e-3,type=float,\
    help='learning rate for the link model')   
parser.add_argument('--nepochs_path',action='store',default=2000,type=int,\
    help='number of epochs for training the path model')
parser.add_argument('--lr_path',action='store',default=1e-3,type=float,\
    help='learning rate for the path model')     
parser.add_argument('--out_var_min',action='store',default=1e-6,type=float,\
    help='min variance in the decoder outputs.  Used for conditioning')     
parser.add_argument('--init_stddev',action='store',default=10.0,type=float,\
    help='weight and bias initialization')
parser.add_argument('--nunits_enc',action='store',nargs='+',\
    default=[100,50],type=int,\
    help='num hidden units for the encoder')    
parser.add_argument('--nunits_dec',action='store',nargs='+',\
    default=[50,100],type=int,\
    help='num hidden units for the decoder')    
parser.add_argument('--nunits_link',action='store',nargs='+',\
    default=[50,25],type=int,\
    help='num hidden units for the link state predictor')        
parser.add_argument('--model_dir',action='store',\
    default='model_data', help='directory to store models')
parser.add_argument('--no_fit_link', dest='no_fit_link', action='store_true',\
    help="Does not fit the link model")
parser.add_argument('--no_fit_path', dest='no_fit_path', action='store_true',\
    help="Does not fit the path model")
parser.add_argument('--checkpoint_period',action='store',default=100,type=int,\
    help='Period in epochs for storing checkpoint.  A value of 0 indicates no checkpoints')    
parser.add_argument('--batch_ind',action='store',default=-1,type=int,\
    help='batch index.  -1 indicates no batch index')    

args = parser.parse_args()
nlatent = args.nlatent
npaths_max = args.npaths_max
nepochs_path = args.nepochs_path
lr_path = args.lr_path
nepochs_link = args.nepochs_link
lr_link = args.lr_link
init_stddev = args.init_stddev
nunits_enc = args.nunits_enc
nunits_dec = args.nunits_dec
nunits_link = args.nunits_link
model_dir = args.model_dir
batch_ind = args.batch_ind
out_var_min = args.out_var_min
fit_link = not args.no_fit_link
fit_path = not args.no_fit_path
checkpoint_period = args.checkpoint_period

# Overwrite parameters based on batch index
lr_batch = [1e-3,1e-3,1e-3,1e-4,1e-4]
nlatent_batch = [10,20,30,20,30]
dir_suffix = ['lr3_nl10', 'lr3_nl20', 'lr3_nl30', 'lr4_nl20', 'lr4_nl30']    
if batch_ind >= 0:
    model_dir = ('model_data_%s' % dir_suffix[batch_ind])
    lr = lr_batch[batch_ind]
    nlatent = nlatent_batch[batch_ind]
    print('batch_ind=%d' % batch_ind)
    print('model_dir= %s' % model_dir)
    print('lr=%12.4e' % lr)
    print('nlatent=%d' % nlatent)
    

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
                   nunits_enc=nunits_enc, nunits_dec=nunits_dec,\
                   nunits_link=nunits_link,\
                   out_var_min=out_var_min,\
                   init_bias_stddev=init_stddev,\
                   init_kernel_stddev=init_stddev, model_dir=model_dir)



if fit_link:
    # Build the link model
    chan_mod.build_link_mod()
    
    # Fit the link model 
    chan_mod.fit_link_mod(train_data, test_data, lr=lr_link,\
                          epochs=nepochs_link)

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
    chan_mod.fit_path_mod(train_data, test_data, lr=lr_path,\
                          epochs=nepochs_path,\
                          checkpoint_period=checkpoint_period)

    
    # Save the path loss model
    chan_mod.save_path_model()




