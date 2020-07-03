"""
create_data.py:  Creates train and test data for the channel modeling
"""
import pandas as pd
import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle


# Parameters
heights = [30, 60, 120]  # Heigths to read
npath_max = 25  # max number of paths per link
ndim = 3        # number of spatial dimensions
pl_max = 200.0  # max path loss.
ts_frac = 0.3   # fraction of samples for test vs. train
tx_pow_dbm = 16 # TX power in dBm

# Create empty arrays on which to stack
nlos_pl = np.zeros((0,npath_max))
los_exists = np.zeros(0,dtype=np.int)
dvec = np.zeros((0,ndim))
los_pl = np.zeros(0)
cell_type = np.zeros(0,dtype=np.int)

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
    
    
    # Get dimensions
    nrx = rx_pos.shape[0]
    ntx = tx_pos.shape[0]
    
    # Get distances
    dvec0 = tx_pos[None,:,:]-rx_pos[:,None,:] 
    
    # Loop over the paths to get data on each tx-rx pair
    los_exists0 = np.zeros((nrx,ntx), dtype=np.int)
    cell_type0 = np.zeros((nrx,ntx), dtype=np.int)
    los_pl0 = np.zeros((nrx,ntx))
    npath_nlos0 = np.zeros((nrx,ntx),dtype=np.int)
    nlos_pl0 = np.tile(pl_max,(nrx,ntx,npath_max))    
    for ipath in range(npath_tot):
        
        irx = rxid[ipath]
        itx = txid[ipath]
        pl_path = tx_pow_dbm - rx_pow_dbm[ipath]
        if los_path[ipath]:
            los_pl0[irx,itx] = pl_path
            los_exists0[irx,itx] = 1        
            
        elif (pl_path < pl_max):
            # Add NLOS path
            j = npath_nlos0[irx,itx]        
            nlos_pl0[irx,itx,j] = pl_path
            npath_nlos0[irx,itx] += 1
        cell_type0[irx,itx]= cell_type_path[ipath]
       
    # Append to arrays
    nlos_pl = np.vstack((nlos_pl, nlos_pl0.reshape((nrx*ntx,npath_max)) ))
    los_exists = np.hstack((los_exists, los_exists0.ravel() ))
    cell_type = np.hstack((cell_type, cell_type0.ravel() ))
    dvec0 = dvec0.reshape((nrx*ntx,ndim))
    dvec = np.vstack((dvec, dvec0))
    los_pl = np.hstack((los_pl, los_pl0.ravel())) 
    
# Split into training and test
ns = dvec.shape[0]
nts = int(ts_frac*ns)
ntr = ns - nts
I = np.random.permutation(ns)
train_data = {'dvec': dvec[I[:ntr]], 'los_exists': los_exists[I[:ntr]], \
              'nlos_pl': nlos_pl[I[:ntr]], 'los_pl':  los_pl[I[:ntr]],\
              'cell_type': cell_type[I[:ntr]]}
test_data = {'dvec': dvec[I[ntr:]], 'los_exists': los_exists[I[ntr:]], \
             'nlos_pl': nlos_pl[I[ntr:]], 'los_pl':  los_pl[I[ntr:]],\
             'cell_type': cell_type[I[ntr:]]}
    

fn = 'train_test_data.p'
with open(fn,'wb') as fp:
    pickle.dump([train_data,test_data,pl_max], fp)    
print('Created file %s' % fn)

        
        
    
    
        




