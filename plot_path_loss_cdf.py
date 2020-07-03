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


from models import ChanMod


# Load the data
fn = 'train_test_data.p'
with open(fn, 'rb') as fp:
    train_data,test_data,pl_max = pickle.load(fp)
    
# Construct the channel model object
K.clear_session()
chan_mod = ChanMod(pl_max=pl_max)

# Load the learned link classifier model
chan_mod.load_link_model()    

# Load the learned path model 
chan_mod.load_path_model()
npaths_max = chan_mod.npaths_max

def comp_pl_omni(pl, pl_max):
    I = np.where(pl < pl_max - 0.1)[0]
    if len(I) == 0:
        pl_omni = pl_max
    else:
        pl_omni = -10*np.log10( np.sum(10**(-0.1*pl[I]) ) )
    return pl_omni

cell_types = [ChanMod.terr_cell, ChanMod.terr_cell,\
              ChanMod.aerial_cell, ChanMod.aerial_cell]
los_types = [1,1,0,0]
title = ['Terr LOS', 'Terr NLOS', 'Aerial LOS', 'Aerial NLOS']
nplot = len(cell_types)


for iplot, cell_type0 in enumerate(cell_types):
        
    # Get the LOS mode 
    los0 = los_types[iplot]
    if los0:
        ls0 = ChanMod.los_link
    else:
        ls0 = ChanMod.nlos_link
    
    # Extract the test links to plot
    dat = test_data
    ls_ts = chan_mod.get_link_state(dat['los_exists'], dat['nlos_pl'])
    
    Its = np.where((ls_ts == ls0) & (dat['cell_type'] == cell_type0))[0]
    
    # Sample from the same conditions
    pl = chan_mod.sample_path(dat['dvec'][Its], dat['cell_type'][Its], \
                              dat['los_exists'][Its])
    
        
    # Get the omni path loss
    ns = len(Its)
    pl_omni_ts = np.zeros(ns)
    pl_omni_rand = np.zeros(ns)
    for i in range(ns):
        pl_omni_ts[i] = comp_pl_omni(dat['nlos_pl'][Its[i],:npaths_max], pl_max)
        pl_omni_rand[i] = comp_pl_omni(pl[i,:npaths_max], pl_max)
    

    # Plot the CDFs
    plt.subplot(2,2,iplot+1)
    p = np.arange(ns)/ns
    plt.plot(np.sort(pl_omni_ts), p)
    plt.plot(np.sort(pl_omni_rand), p)
    plt.grid()
    plt.title(title[iplot])
    plt.legend(['Test', 'VAE'])
    if (iplot==0) or (iplot==2):
        plt.ylabel('CDF')
    if (iplot==2) or (iplot==3):
        plt.xlabel('Path loss (dB)')
    plt.xlim([np.min(pl_omni_ts), np.max(pl_omni_ts)])

plt.tight_layout()

     
plt.savefig('omni_path_loss.png', bbox_inches='tight')





