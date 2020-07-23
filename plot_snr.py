"""
plot_link_mod:  Link model plot
"""
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers
import tensorflow.keras.backend as K

from tqdm import trange
from models import ChanMod, DataFormat
from antenna import Elem3GPP, Elem3GPPMultiSector

# Paramters
bw = 400e6   # Bandwidth in Hz
nf = 6  # Noise figure in dB
bf_gain = 10*np.log10(64*16)-3  # Array gain 
kT = -174   # Thermal noise in dB/Hz
tx_pow = 23  # TX power in dBm
npts = 100   # number of points for each (x,z) bin
aer_height=30  # Height of the aerial cell
downtilt = 10  # downtilt on terrestrial cells

# Number of x and z bins
nx = 40
nz = 20

# Range of x and z distances to test
xlim = np.array([0,500])
zlim = np.array([0,130])
        
def comp_pl_gain(pl, gain, pl_max):
    """
    Computes the path loss with directional gain

    Parameters
    ----------
    pl : (nlink,npaths_max) array
        Array of path losses for each path
    gain : (nlink, npaths_max) array
        Array of gains on each path
    pl_max : scalar
        Max path loss.  Values below this are not considerd

    Returns
    -------
    pl_dir : (nlink,) array
        Effective path loss with directional gain

    """
    n = pl.shape[0]
    pl_dir = np.zeros(n)
    for i in range(n):
        I = np.where(pl[i,:] < pl_max - 0.1)[0]
        if len(I) == 0:
            pl_dir[i] = pl_max
        else:
            pl_gain = pl[i,I] - gain[i,I]
            pl_dir[i] = -10*np.log10( np.sum(10**(-0.1*pl_gain) ) )
    return pl_dir
 
 
# Construct the channel model object
model_dir = 'model_data'

K.clear_session()
chan_mod = ChanMod(model_dir=model_dir)

# Load the learned link classifier model
chan_mod.load_link_model()    

# Load the learned path model 
chan_mod.load_path_model(weights_fn='path_weights.h5')
pl_max = chan_mod.pl_max

cell_types = [ChanMod.terr_cell, ChanMod.aerial_cell]
cell_str = ['Terestrial', 'Aerial']

nplot = len(cell_types)
snr_med = np.zeros((nz,nx,nplot))

for iplot, cell_type0 in enumerate(cell_types):
    
    # Print cell type
    print('Simulating %s cell' % cell_str[iplot])
    
    # Set the limits and x and z values to test
    dx = np.linspace(xlim[0],xlim[1],nx)        
    dz = np.linspace(zlim[0],zlim[1],nz)
    if cell_type0 == ChanMod.aerial_cell:
        dz = dz - aer_height
    
    # Create the BS elements
    if cell_type0 == ChanMod.aerial_cell:
        elem_gnb = Elem3GPP(theta0=0, thetabw=90, phibw=0)
    else:
        elem_gnb = Elem3GPPMultiSector(nsect=3,theta0=90+downtilt,\
                                       thetabw=65,\
                                       phi0=0, phibw=90)
    elem_ue = Elem3GPP(theta0=180, thetabw=90, phibw=0)
    
    
    # Convert to meshgrid
    dxmat, dzmat = np.meshgrid(dx,dz)
    
    dvec = np.zeros((nx*nz,3))
    dvec[:,0] = dxmat.ravel()
    dvec[:,2] = dzmat.ravel()
    cell_type = np.tile(cell_type0, (nx*nz,))
    
    
    # Loop over multiple trials
    snr = np.zeros((nz,nx,npts))
    
    for i in trange(npts):
        # Generate random channels
        pl, ang = chan_mod.sample_path(dvec, cell_type) 
            
        # Compute the UE gain
        ue_theta = ang[:,:,DataFormat.aoa_theta_ind]
        ue_phi = ang[:,:,DataFormat.aoa_phi_ind]
        gain_ue = elem_ue.response(ue_phi, ue_theta)
        
        # Compute the gNB gain
        gnb_theta = ang[:,:,DataFormat.aod_theta_ind]
        gnb_phi = ang[:,:,DataFormat.aod_phi_ind]
        gain_gnb = elem_ue.response(gnb_phi, gnb_theta)
        gain_tot = gain_ue + gain_gnb
        
        # Compute the directional gain and RX power
        pl_gain = comp_pl_gain(pl, gain_tot, pl_max)
        snri = tx_pow - pl_gain + bf_gain - kT - nf - 10*np.log10(bw)
    
        # Create the data for the plot    
        snri = snri.reshape((nz,nx))
        snri = np.flipud(snri)
        
        snr[:,:,i] = snri
        #print('iplot=%d i=%d' % (iplot, i))
     
    # Get the median rx power
    snr_med[:,:,iplot] = np.median(snr,axis=2) 
 
       
# Plot the results
for iplot in range(nplot):
                    
    plt.subplot(1,nplot,iplot+1)
    plt.imshow(snr_med[:,:,iplot],\
               extent=[np.min(xlim),np.max(xlim),np.min(zlim),np.max(zlim)],\
               aspect='auto', vmin=-20, vmax=40)   
        
    # Add horizontal line indicating location of aerial cell
    if (cell_types[iplot] == ChanMod.aerial_cell):
        plt.plot(xlim, np.array([1,1])*aer_height, 'r--')
        
    if (iplot > 0):
        plt.yticks([])
    else:
        plt.ylabel('Elevation (m)')
    plt.xlabel('Horiz (m)')
    plt.title(cell_str[iplot])
        

# Add the colorbar
plt.tight_layout()
plt.subplots_adjust(bottom=0.1, right=0.87, top=0.9)
cax = plt.axes([0.92, 0.1, 0.05, 0.8])
plt.colorbar(cax=cax)        

plt.savefig('snr_dist.png', bbox_inches='tight')
    
    
    


