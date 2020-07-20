"""
plot_angle_dist.py:  Plots the angular distribution

For all the NLOS paths, the program:
* Computes the  AoA and AoD relative to the LOS path
* Plots the empirical distribution of the relative angles as 
  a function of the distance
* Generates random angles with the same conditions as the model,
  and plots the relative angle as a function of the distance
  for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import models
import tensorflow.keras.backend as K
from models import ChanMod, DataFormat, get_link_state

def plot_ang_dist(chan_mod,dvec,nlos_ang,nlos_pl,iang,np_plot=10):
    """
    Plots the conditional distribution of the relative angle.
    
    Parameters
    ----------
    chan_mod : ChanMod structure
        Channel model.
    dvec : (nlink,ndim) array
            vector from cell to UAV
    nlos_ang : (nlink,npaths_max,nangle) array
            Angles of each path in each link.  
            The angles are in degrees
    nlos_pl : (nlink,npaths_max) array 
            Path losses of each path in each link.
            A value of pl_max indicates no path
    iang: integer from 0 to DataFormat.nangle-1
        Index of the angle to be plotted
    np_plot:  integer
        Number of paths whose angles are to be plotted
    """
    # Get the distances
    np_plot = 10
    dist = np.sqrt(np.sum(dvec**2,axis=1))
    dist_plot = np.tile(dist[:,None],(1,np_plot))
    dist_plot = dist_plot.ravel()
    
    # Transform the angles.  The transformations compute the
    # relative angles and scales them by 180
    ang_tr = chan_mod.transform_ang(dvec, nlos_ang, nlos_pl)
    
    # Get the relative angle
    np_max = chan_mod.npaths_max    
    ang_rel = ang_tr[:,iang*np_max:iang*np_max+np_plot]*180
    ang_rel = ang_rel.ravel()
    
    # Set the angle and distance range for the historgram
    drange = [0,600]
    if iang==DataFormat.aoa_phi_ind or iang==DataFormat.aod_phi_ind:
        ang_range = [-180,180]
    elif iang==DataFormat.aoa_theta_ind or iang==DataFormat.aod_theta_ind:
        ang_range = [-90,90]
    else:
        raise ValueError('Invalid angle index')
    
    # Compute the emperical conditional probability
    H0, dedges, ang_edges = np.histogram2d(dist_plot,ang_rel,bins=[10,40],\
                                           range=[drange,ang_range])       
    Hsum = np.sum(H0,axis=1)
    H0 = H0 / Hsum[:,None]
    
    # Plot the log probability.
    # We plot the log proability since the probability in linear
    # scale is difficult to view
    log_prob = np.log10(np.maximum(0.01,H0.T))
    plt.imshow(log_prob, extent=[np.min(dedges),np.max(dedges),\
               np.min(ang_edges),np.max(ang_edges)], aspect='auto')   


"""
Load the true data
"""

# Load the data
fn = 'train_test_data.p'
with open(fn, 'rb') as fp:
    train_data,test_data,pl_max = pickle.load(fp)
    

# Get the variables
dat = test_data

# Find the links where there is at least one valid path (either LOS or NLOS)
chan_mod0 = ChanMod()
ls = get_link_state(dat['los_exists'], dat['nlos_pl'], pl_max)
Ilink = np.where(ls != ChanMod.no_link)[0]
    
# Get the data from these links
dvec = dat['dvec'][Ilink]
los_ang = dat['los_ang'][Ilink]
nlos_ang = dat['nlos_ang'][Ilink]
nlos_pl = dat['nlos_pl'][Ilink]
ls = ls[Ilink]

"""
Generate synthentic data from trained model
"""

# Model directory
model_dir = 'models_20200719/model_data_lr4_nl10_mv4'

# Construct the channel model object
K.clear_session()
chan_mod = ChanMod(pl_max=pl_max,model_dir=model_dir)    

# Load the learned link classifier model
chan_mod.load_link_model()    

# Load the learned path model 
chan_mod.load_path_model()

# Sample from the same conditions as the data
nlos_pl_rand, nlos_ang_rand = chan_mod.sample_path(dvec,\
        dat['cell_type'][Ilink], ls, nlos_only=True)
    
"""
Plot the angular distributions
"""    
plt.figure(figsize=[10,10])
ang_str = ['AoA Az', 'AoA El', 'AoD Az', 'AoD El']
for iang in range(4):
    plt.subplot(4,2,2*iang+1)
    plot_ang_dist(chan_mod0,dvec,nlos_ang,nlos_pl,iang)
    if iang < 3:
        plt.xticks([])
    else:
        plt.xlabel('Dist (m)')
    title_str = ang_str[iang] + ' Data'
    plt.title(title_str)
    
    plt.subplot(4,2,2*iang+2)
    plot_ang_dist(chan_mod,dvec,nlos_ang_rand,nlos_pl_rand,iang)
    if iang < 3:
        plt.xticks([])
    else:
        plt.xlabel('Dist (m)')        
    title_str = ang_str[iang] + ' Model'
    plt.title(title_str)
    
plt.tight_layout()


if 0:            
    plt.subplots_adjust(bottom=0.1, right=0.87, top=0.9)
    cax = plt.axes([0.92, 0.1, 0.05, 0.8])
    plt.colorbar(cax=cax)
    
plt.savefig('angle_dist.png')

    


