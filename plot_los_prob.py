"""
plot_link_mod:  Link model plot
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
from sklearn.neighbors import KNeighborsClassifier


# Load the data
fn = 'train_test_data.p'
with open(fn, 'rb') as fp:
    train_data,test_data,pl_max = pickle.load(fp)
 
# Construct the channel model object
K.clear_session()
chan_mod = ChanMod()

# Load the link classifier model
chan_mod.load_link_model()
    
cell_types = [ChanMod.aerial_cell, ChanMod.terr_cell]
title = ['Aerial', 'Terrestrial']
nplot = len(cell_types)
link_state = ChanMod.los_link

for i, cell_type0 in enumerate(cell_types):
    """
    Train a non-parametric classifier
    """
 
    # Get the test data vector
    dat = test_data
    dvec = dat['dvec']
    dx = np.sqrt(dvec[:,0]**2 + dvec[:,1]**2)
    dz = dvec[:,2]
    xscale = 0.2

    # Get the link state
    ls_ts = chan_mod.get_link_state(dat['los_exists'], dat['nlos_pl'])
    
    # Create a classifier
    clf = KNeighborsClassifier(20)
    
    # Fit the classifier    
    X = np.column_stack((dz,xscale*dx))
    clf.fit(X,ls_ts)
    
  

    """
    Plot the learned model
    """
    # Get the horiz and elevation distances to plot
    nx = 40
    nz = 20

    if cell_type0 == ChanMod.aerial_cell:
        zlim = np.array([-80,130])        
    else:
        zlim = np.array([-10,130])
    dx = np.linspace(1,500,nx)        
    dz = np.linspace(zlim[0],zlim[1],nz)
    dxmat, dzmat = np.meshgrid(dx,dz)
    
    for j in range(2):

        if j==1:    
            X = np.column_stack((dzmat.ravel(), xscale*dxmat.ravel()))
            prob = clf.predict_proba(X)
        
        else:    
            # Run the model to predict the probabilities
            dvec = np.zeros((nx*nz,3))
            dvec[:,0] = dxmat.ravel()
            dvec[:,2] = dzmat.ravel()
            cell_type = np.tile(cell_type0, (nx*nz,))
            prob = chan_mod.link_predict(dvec, cell_type)
        #psum = (prob[:,1]+prob[:,2])
        #prob = prob/psum[:,None]
        
        # Plot the probabilties
        prob = prob.reshape((nz,nx,ChanMod.nlink_states))
        plt.subplot(2,2,i+2*j+1)
        plt.imshow(prob[:,:,link_state],\
                   extent=[np.min(dz),np.max(dz),np.min(dx),np.max(dx)],\
                   aspect='auto', vmin=0, vmax=1)   
        plt.xlim(zlim[0], zlim[1])
        if (i > 0):
            plt.yticks([])
        else:
            plt.ylabel('Horiz (m)')
            
        # Print the title
        if j == 0:
            plt.title(title[i] + ' NN')
        else:
            plt.title(title[i] + ' KNN')



if 1:
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, right=0.87, top=0.9)
    cax = plt.axes([0.92, 0.1, 0.05, 0.8])
    plt.colorbar(cax=cax)


