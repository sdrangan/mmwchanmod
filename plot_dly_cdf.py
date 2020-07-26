"""
plot_dly_cdf:  Plots the CDF of the delay spread on the test data,
and compares that to the randomly generated path loss from the trained model.

For the test data, we have (Uts,Xts) which is the link condition and 
path data.  We then generate synthetic samples, with the same condition 
using the trained VAE.  That is,

    Xrand = g(Uts,Zrand)   Zrand ~ N(0,I)
    
where is the g(.) is the conditional VAE.

Xrand and Xts are both vectors for each sample.  We compute statistics,
dly_rms_ts and dly_rms_rand, the RMS delay spreads on the random
and test data.  We then plot the CDFs of the two and compare.
    
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers
import tensorflow.keras.backend as K


from models import ChanMod, combine_los_nlos, get_link_state

def comp_dly_rms(dly, pl, pl_max):
    """
    Computes rms delay spread

    Parameters
    ----------
    dly : (nlink,npaths_max) array
        Delay of the paths in the link in seconds
    pl : (nlink,npaths_max) array
        Path loss of the paths in the link in seconds
    pl_max : scalar
        Max path loss.  Values above this are not considered

    Returns
    -------
    dly_rms:  (nlink,) array
        RMS delay spread of the links in seconds

    """
    
    # Initialize array
    nlink = pl.shape[0]
    dly_rms = np.zeros(nlink)
    
    # Loop over links
    for i in range(nlink):
        # Find invalid paths
        I = np.where(pl[i,:] < pl_max - 0.1)[0]
        if len(I) > 0:
            pl_lin = 10**(-0.1*pl[i,I])
            Z = np.sum(pl_lin)
            dly_mean = dly[i,I].dot(pl_lin)/Z
            dly_sq = (dly[i,I]-dly_mean)**2
            dly_rms[i] = np.sqrt(dly_sq.dot(pl_lin)/Z)
    return dly_rms

model_dir = 'model_data'


# Load the data
fn = 'train_test_data.p'
with open(fn, 'rb') as fp:
    train_data,test_data,pl_max = pickle.load(fp)
    
# Combine the LOS and NLOS path data
data = test_data
pl_dat, ang_dat, dly_dat = combine_los_nlos(\
    data['nlos_pl'], data['nlos_ang'], data['nlos_dly'], data['los_exists'],\
    data['los_pl'], data['los_ang'], data['los_dly'] )
    
# Get the link state
link_state = get_link_state(data['los_exists'], data['nlos_pl'], pl_max)
    
# Construct the channel model object
K.clear_session()
chan_mod = ChanMod(pl_max=pl_max,model_dir=model_dir)

# Load the learned link classifier model
chan_mod.load_link_model()    

# Load the learned path model 
chan_mod.load_path_model(weights_fn='path_weights.h5')
npaths_max = chan_mod.npaths_max

# Sample from the same conditions
pl_rand, ang_rand, dly_rand = chan_mod.sample_path(data['dvec'],\
        data['cell_type'], link_state)
    

    
cell_types = [ChanMod.terr_cell, ChanMod.aerial_cell]
title = ['Terrestrial', 'Aerial']
nplot = len(cell_types)
for iplot, cell_type0 in enumerate(cell_types):

    I = np.where((link_state != ChanMod.no_link)\
                 & (data['cell_type'] == cell_type0))[0]
        
    dly_rms_dat = comp_dly_rms(dly_dat[I], pl_dat[I], pl_max)
    dly_rms_rand = comp_dly_rms(dly_rand[I], pl_rand[I], pl_max)
    
    nlink = len(dly_rms_dat)
    p = np.arange(nlink)/nlink
    
    plt.subplot(1,nplot,iplot+1)
    plt.plot(np.sort(dly_rms_dat)*1e9, p)
    plt.plot(np.sort(dly_rms_rand)*1e9, p)
    plt.xlim([0,800])
    plt.title(title[iplot])
    plt.legend(['Data', 'Model'], loc='center right')
    if iplot == 0:
        plt.ylabel('CDF')            
    plt.grid()        
    plt.xlabel('RMS delay spread (ns)')
    
plt.tight_layout()
plt.savefig('dly_rms.png', bbox_inches='tight')
        
    
