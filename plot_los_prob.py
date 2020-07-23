"""
plot_los_prob.py:  Plots the LOS probability as a function of the position
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers
import tensorflow.keras.backend as K

from models import ChanMod, get_link_state


# Load the data
fn = 'train_test_data.p'
with open(fn, 'rb') as fp:
    train_data,test_data,pl_max = pickle.load(fp)
 
# Construct the channel model object
K.clear_session()
chan_mod = ChanMod()

# Load the link classifier model
chan_mod.load_link_model()
    
# Plot paramters
cell_types = [ChanMod.aerial_cell,ChanMod.terr_cell]
title = ['Aerial', 'Terrestrial']
nplot = len(cell_types)
ls0 = ChanMod.los_link

for i, cell_type0 in enumerate(cell_types):
    
 
    """
    Compute the histogram on the test data
    """
    # Set the limits
    xlim = np.array([0,500])
    if cell_type0 == ChanMod.aerial_cell:
        zlim = np.array([-80,130])        
    else:
        zlim = np.array([0,130])    
    
    # Get the test data vector
    dat = test_data
    dvec = dat['dvec']
    dx = np.sqrt(dvec[:,0]**2 + dvec[:,1]**2)
    dz = dvec[:,2]
    

    # Get the link state
    ls_ts = get_link_state(dat['los_exists'], dat['nlos_pl'], pl_max)
    
    # Extract the correct points    
    I = np.where(dat['cell_type'] == cell_type0)[0]
    I0 = np.where((dat['cell_type'] == cell_type0) & (ls_ts==ls0))[0]
    
    
    # Compute the empirical probability
    H0, xedges, zedges = np.histogram2d(dx[I0],dz[I0],bins=[10,5],range=[xlim,zlim])
    Htot, xedges, zedges = np.histogram2d(dx[I],dz[I],bins=[10,5],range=[xlim,zlim])
    prob_ts = H0 / np.maximum(Htot,1)
    prob_ts = np.flipud(prob_ts.T)
    
    # Plot the results
    plt.subplot(2,2,2*i+1)
    plt.imshow(prob_ts,aspect='auto',\
               extent=[np.min(xedges),np.max(xedges),np.min(zedges),np.max(zedges)],\
               vmin=0, vmax=1)
    plt.title(title[i] + ' Data')
    plt.ylabel('Elev (m)')
    if (i == 0):
        plt.xticks([])
    else:
        plt.xlabel('Horiz (m)')        
      

    """
    Plot the learned model
    """
    # Get the horiz and elevation distances to plot
    nx = 40
    nz = 20

    
    dx = np.linspace(xlim[0],xlim[1],nx)        
    dz = np.linspace(zlim[0],zlim[1],nz)
    dxmat, dzmat = np.meshgrid(dx,dz)
    
    
    # Run the model to predict the probabilities
    dvec = np.zeros((nx*nz,3))
    dvec[:,0] = dxmat.ravel()
    dvec[:,2] = dzmat.ravel()
    cell_type = np.tile(cell_type0, (nx*nz,))
    prob = chan_mod.link_predict(dvec, cell_type)
        
    # Plot the probabilties
    prob = prob.reshape((nz,nx,ChanMod.nlink_states))
    prob = np.flipud(prob[:,:,ls0])
    plt.subplot(2,2,2*i+2)
    plt.imshow(prob,\
               extent=[np.min(dx),np.max(dx),np.min(dz),np.max(dz)],\
               aspect='auto', vmin=0, vmax=1)   
    plt.xlim(xlim[0], xlim[1])
    if (i ==  0):
        plt.xticks([])
    else:
        plt.xlabel('Horiz (m)')
            
    # Print the title
    plt.title(title[i] + ' Model')
            



if 1:
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, right=0.87, top=0.9)
    cax = plt.axes([0.92, 0.1, 0.05, 0.8])
    plt.colorbar(cax=cax)

plt.savefig('los_prob.png', bbox_inches='tight')
