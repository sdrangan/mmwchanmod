"""
plot_path_loss_cdf:  Plots the CDF of the path loss on the test data,
and compares that to the randomly generated path loss from the trained model.

For the test data, we have (Uts,Xts) which is the link condition and 
path data.  We then generate synthetic samples, with the same condition 
using the trained VAE.  That is,

    Xrand = g(Uts,Zrand)   Zrand ~ N(0,I)
    
where is the g(.) is the conditional VAE.

Xrand and Xts are both vectors for each sample.  We compute a statistic

    pl_omni_ts = omni(Xts)  = total omni path loss
    pl_omni_rand = omni(Xrand)  = total omni path loss
    
Now we have a set of scalar.  We plot the CDF of pl_omni_ts and pl_omni_rand
and compare.
    
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

def comp_pl_omni(pl, pl_max):
    I = np.where(pl < pl_max - 0.1)[0]
    if len(I) == 0:
        pl_omni = pl_max
    else:
        pl_omni = -10*np.log10( np.sum(10**(-0.1*pl[I]) ) )
    return pl_omni

model_dir = 'model_data'


# Load the data
fn = 'train_test_data.p'
with open(fn, 'rb') as fp:
    train_data,test_data,pl_max = pickle.load(fp)
    
# Combine the LOS and NLOS path data
data = test_data
pl_dat, ang_dat, pl_dly = combine_los_nlos(\
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


plt.rcParams.update({'font.size': 14})    
cell_types = [ChanMod.terr_cell, ChanMod.aerial_cell]
title = ['Terrestrial', 'Aerial']
for iplot, cell_type0 in enumerate(cell_types):
    I = np.where((link_state != ChanMod.no_link)\
                 & (data['cell_type'] == cell_type0))[0]
        
    # Get the omni path loss
    ns = len(I)
    pl_omni_dat = np.zeros(ns)
    pl_omni_rand = np.zeros(ns)
    for i in range(ns):
        pl_omni_rand[i]  = comp_pl_omni(pl_rand[I[i],:npaths_max], pl_max)
        pl_omni_dat[i] = comp_pl_omni(pl_dat[I[i],:npaths_max], pl_max)
    
    # Plot the CDFs
    plt.subplot(1,2,iplot+1)
    p = np.arange(ns)/ns
    plt.plot(np.sort(pl_omni_dat), p)
    plt.plot(np.sort(pl_omni_rand), p)    
    plt.title(title[iplot])
    plt.legend(['Data', 'Model'])
    if iplot == 0:
        plt.ylabel('CDF')    
    plt.grid()        
    plt.xlabel('Path loss (dB)')
    
        
        
plt.tight_layout()
plt.savefig('omni_path_loss.png', bbox_inches='tight')


