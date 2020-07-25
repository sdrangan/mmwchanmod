"""
create_data.py:  Creates train and test data for the channel modeling
"""
import pandas as pd
import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from models import DataFormat


# Parameters
heights = [30, 60, 120]  # Heigths to read
npath_max = 25  # max number of paths per link
ndim = 3        # number of spatial dimensions
pl_max = 200.0  # max path loss.
ts_frac = 0.3   # fraction of samples for test vs. train
tx_pow_dbm = 16 # TX power in dBm

# Angles will be stored as [aoa_az,aoa_el,aod_az,aod_el]
nangle = DataFormat.nangle

# Create empty arrays on which to stack
nlos_pl = np.zeros((0,npath_max))
nlos_dly = np.zeros((0,npath_max))
los_exists = np.zeros(0,dtype=np.int)
dvec = np.zeros((0,ndim))
los_pl = np.zeros(0)
los_dly = np.zeros(0)
cell_type = np.zeros(0,dtype=np.int)
nlos_ang = np.zeros((0,npath_max,nangle))
los_ang = np.zeros((0,nangle))
ang = np.zeros((0,npath_max,nangle))

# Loop over the heights to extract the data
for height in heights:
    
    # Display progress
    matlab_dir = '..\Remcom_Channel_Stats\%dm' % height
    print('Processing directory %s' % matlab_dir)
    
    # Load the path table
    filename = 'paths%d.csv' % height
    df = pd.read_table(filename, sep=',')
    
    # Load the RX and TX locations    
    fn = matlab_dir + os.path.sep + 'RX_locations.mat'
    mat = scipy.io.loadmat(fn)
    rx_pos = mat['rxpos']
    fn = matlab_dir + os.path.sep + 'TX_locations.mat'
    mat = scipy.io.loadmat(fn)
    tx_pos = mat['txpos']
    
    # Get the fields from the tables    
    npath_tot = df.shape[0]
    txid = df['UAV_ID'].to_numpy()-1
    rxid = df['RXID'].to_numpy()-1
    cell_type_path = df['RX_Type'].to_numpy()-1
    los_path = df['los'].to_numpy()
    rx_pow_dbm = df['rx_power_dbm'].to_numpy()    
    ang_path = df[['aoa_phi_deg','aoa_theta_deg',\
                   'doa_phi_deg','doa_theta_deg']].to_numpy()
    toa_path = df['toa_sec'].to_numpy()
        
    # Get dimensions
    nrx = rx_pos.shape[0]
    ntx = tx_pos.shape[0]
    
    # Get distances
    dvec0 = tx_pos[None,:,:]-rx_pos[:,None,:] 
    
    # Initialize the vectors to hold the path data
    los_exists0 = np.zeros((nrx,ntx), dtype=np.int)
    cell_type0 = np.zeros((nrx,ntx), dtype=np.int)
    los_pl0 = np.zeros((nrx,ntx))
    los_dly0 = np.zeros((nrx,ntx))
    npath_nlos0 = np.zeros((nrx,ntx),dtype=np.int)
    nlos_pl0 = np.tile(pl_max,(nrx,ntx,npath_max))    
    nlos_dly0 = np.zeros((nrx,ntx,npath_max)) 
    los_ang0 = np.zeros((nrx,ntx,nangle))
    nlos_ang0 = np.zeros((nrx,ntx,npath_max,nangle))
    ang0 = np.zeros((nrx,ntx,npath_max,nangle))
  
    itx_prev = -1
    irx_prev = -1
    pathNum = 0
    # Loop over the paths to get data on each tx-rx pair
    for ipath in range(npath_tot):
    
        irx = rxid[ipath]
        itx = txid[ipath]
        pl_path = tx_pow_dbm - rx_pow_dbm[ipath]
        
        if los_path[ipath]:
            # LOS path case.  
            los_pl0[irx,itx] = pl_path
            los_dly0[irx,itx] = toa_path[ipath]
            los_ang0[irx,itx,:] = ang_path[ipath,:]
            los_exists0[irx,itx] = 1        
            
        elif (pl_path < pl_max):
            # NLOS path case
            j = npath_nlos0[irx,itx]        
            nlos_pl0[irx,itx,j] = pl_path
            nlos_dly0[irx,itx,j] = toa_path[ipath]
            npath_nlos0[irx,itx] += 1 
            nlos_ang0[irx,itx,j,:] = ang_path[ipath,:]
        
        
        cell_type0[irx,itx,] = cell_type_path[ipath]
                
        
    # Append to arrays
    nlos_pl = np.vstack((nlos_pl, nlos_pl0.reshape((nrx*ntx,npath_max)) ))
    nlos_dly = np.vstack((nlos_dly, nlos_dly0.reshape((nrx*ntx,npath_max)) ))
    los_exists = np.hstack((los_exists, los_exists0.ravel() ))
    cell_type = np.hstack((cell_type, cell_type0.ravel() ))
    dvec0 = dvec0.reshape((nrx*ntx,ndim))
    dvec = np.vstack((dvec, dvec0))
    los_pl = np.hstack((los_pl, los_pl0.ravel())) 
    los_dly = np.hstack((los_dly, los_dly0.ravel())) 
    los_ang = np.vstack((los_ang, los_ang0.reshape((nrx*ntx,nangle)) ))
    nlos_ang = np.vstack((nlos_ang,\
                          nlos_ang0.reshape((nrx*ntx,npath_max,nangle)) ))

    
# Split into training and test
ns = dvec.shape[0]
nts = int(ts_frac*ns)
ntr = ns - nts
I = np.random.permutation(ns)
train_data = {'dvec': dvec[I[:ntr]], 'los_exists': los_exists[I[:ntr]], \
              'nlos_pl': nlos_pl[I[:ntr]], 'los_pl':  los_pl[I[:ntr]],\
              'cell_type': cell_type[I[:ntr]], 'nlos_ang': nlos_ang[I[:ntr]],\
              'los_ang': los_ang[I[:ntr]], 'nlos_dly': nlos_dly[I[:ntr]],\
              'los_dly': los_dly[I[:ntr]] }
test_data = {'dvec': dvec[I[ntr:]], 'los_exists': los_exists[I[ntr:]], \
             'nlos_pl': nlos_pl[I[ntr:]], 'los_pl':  los_pl[I[ntr:]],\
             'cell_type': cell_type[I[ntr:]], 'nlos_ang': nlos_ang[I[ntr:]],\
              'los_ang': los_ang[I[ntr:]], 'nlos_dly': nlos_dly[I[ntr:]],\
              'los_dly': los_dly[I[ntr:]]}
    
if 1:
    fn = 'train_test_data.p'
    with open(fn,'wb') as fp:
        pickle.dump([train_data,test_data,pl_max], fp)    
    print('Created file %s' % fn)

        
        
    
    
        




