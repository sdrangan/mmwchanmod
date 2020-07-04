"""
vae.py:  Conditional VAE model 

"""
import tensorflow as tf
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers
import tensorflow.keras.backend as K
import numpy as np
import sklearn.preprocessing
import pickle
from tensorflow.keras.optimizers import Adam
import os


class CondVAE(object):
    def __init__(self, nlatent, ndat, ncond, nunits_enc=(25,10,),\
                 nunits_dec=(10,25,), out_var_min=1e-4,\
                 init_kernel_stddev=10.0, init_bias_stddev=10.0,\
                 sort_out=False,**kwargs):
        """
        Conditional VAE class

        Parameters
        ----------
        nlatent : int
            number of latent states
        ndat : int
            number of features in the data to be modeled
        ncond : int
            number of conditional variables
        nunits_enc : list of integers
            number of hidden units in each layer of the encoder
        nunits_dec : list of integers
            number of hidden units in each layer of the decoder
        sort_out : boolean
            flag indicating if output mean is sorted 
            This is used for the path loss data
        out_var_min:  scalar
            minimum output variance.  This is used for improved conditioning
        init_kernel_stddev : scalar
            std deviation of the kernel in the initialization
        init_bias_stddev : scalar
            std deviation of the bias in the initialization

        """   
        self.nlatent = nlatent
        self.ndat = ndat        
        self.ncond = ncond
        self.nunits_enc = nunits_enc
        self.out_var_min = out_var_min
        self.sort_out = sort_out
        self.init_kernel_stddev = init_kernel_stddev
        self.init_bias_stddev = init_bias_stddev
        
        self.build_encoder()
        self.build_decoder()
        self.build_vae()
        

    def build_encoder(self):
        """
        Builds the encoder network
        
        The encoder maps [x,cond] to [z_mu, z_log_var]
        """
        x = tfkl.Input((self.ndat,), name='x')
        cond = tfkl.Input((self.ncond,), name='cond')
        
        dat_cond = tfkl.Concatenate(name='dat_cond')([x, cond])
        
        
        
        # Add the hidden layers
        h = dat_cond
        layer_names = []
        for i in range(len(self.nunits_enc)):           
            h = tfkl.Dense(self.nunits_enc[i], activation='sigmoid',\
                           name='FC%d' % i)(h)
            layer_names.append('FC%d' % i)
            
        # Add the final output layer                
        z_mu = tfkl.Dense(self.nlatent, activation='linear',\
                          bias_initializer=None, name='z_mu')(h)
        z_log_var = tfkl.Dense(self.nlatent, activation='linear',\
                          bias_initializer=None, name='z_log_var')(h)
                
        # Save the encoder model
        self.encoder = tfkm.Model([x, cond],\
                                  [z_mu, z_log_var])

        # Set the initialization
        set_initialization(self.encoder, layer_names,\
                           self.init_kernel_stddev, self.init_bias_stddev)        
            
    def reparm(self, z_mu, z_log_var):
        """
        Re-parametrization layer
        
            z = z_mu + eps * tf.exp(z_log_var*0.5)
            
        where eps is unit Gaussian
        """
        batch_shape = tf.shape(z_mu)
        eps = tf.random.normal(shape=batch_shape)
        z = eps * tf.exp(z_log_var*0.5) + z_mu
        return z
        
        
    def build_decoder(self):
        """
        Builds the decoder network.
        The decoder network is the generative model mapping:
            
            [z,cond] to xhat

        """   
        # Input layer
        z_samp = tfkl.Input((self.nlatent,), name='z')
        cond = tfkl.Input((self.ncond,), name='cond')   
        z_cond = tfkl.Concatenate(name='z_cond')([z_samp, cond])
        
        # Hidden layers
        layer_names = []
        h = z_cond
        for i in range(len(self.nunits_enc)):            
            h = tfkl.Dense(self.nunits_enc[i], activation='sigmoid',\
                           bias_initializer=None, name='FC%d' % i)(h)
            layer_names.append('FC%d' % i)
            
        # Add the output mean with optional sorting        
        x_mu = tfkl.Dense(self.ndat, name='x_mu',\
                          bias_initializer=None)(h)
        if self.sort_out:
            x_mu = tf.sort(x_mu, direction='DESCENDING', axis=-1)
                                
        # Add the output variance.                            
        x_log_var = tfkl.Dense(self.ndat, name='x_log_var')(h)   
        x_log_var = tf.maximum(x_log_var, np.log(self.out_var_min) )            
        
        # Build the decoder
        self.decoder = tfkm.Model([z_samp, cond], [x_mu, x_log_var])
        
        # Set the initialization
        set_initialization(self.decoder, layer_names,\
                           self.init_kernel_stddev, self.init_bias_stddev)      
        
        # Build the decoder with sampling
        x_samp = self.reparm(x_mu, x_log_var)
        self.sampler = tfkm.Model([z_samp, cond], x_samp)
        
        
    def build_vae(self):
        """
        Builds the VAE to train.  
        
        The VAE takes an input sample x and outputs [xhat,x_log_var].
        It also has the reconstruction and KL divergence loss
        
        """
        # Build the encoder and decoder
        self.build_encoder()
        self.build_decoder()
        
        # Inputs for the VAE
        x = tfkl.Input((self.ndat,), name='x')
        cond = tfkl.Input((self.ncond,), name='cond')
        
        # Encoder
        z_mu, z_log_var = self.encoder([x,cond])
        z_samp = self.reparm(z_mu, z_log_var)
        
        # Decoder
        xhat, x_log_var = self.decoder([z_samp, cond])
        self.vae = tfkm.Model([x, cond], [xhat, x_log_var])
            
        # Add reconstruction loss   
        recon_loss = K.square(xhat - x)*tf.exp(-x_log_var) + x_log_var + \
            np.log(2*np.pi)
        recon_loss = 0.5*K.sum(recon_loss, axis=-1)
        
        # Add the KL divergence loss
        kl_loss = 1 + z_log_var - K.square(z_mu) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(recon_loss + kl_loss)
        self.vae.add_loss(vae_loss)

class ChanMod(object):
    """
    Object for modeling mmWave channel model data.
    
    There are two parts in the model:
        * link_mod:  This predicts the link_state (i.e. LOS, NLOS or no link)
          from the link conditions.  This is implemented a neural network
        * path_mod:  This predicts the other channel parameters (right now,
          this is the vector of path losses) from the condition and link_state.
        
    Each model has a pre-processor on the data and conditions that is also
    trained.
          
    """
    
    """
    Static variables
    """
    # Link states
    no_link = 0
    los_link = 1
    nlos_link = 2
    nlink_states = 3
    
    # Cell types
    terr_cell = 0
    aerial_cell = 1
    ncell_type = 2
    
    # Numbers of transformed features for models
    nin_link = 5   # num features for link predictor model
    ncond = 5      # num condition features for path model
    
    
    
    def __init__(self,npaths_max=25,pl_max=200, nlatent=10,\
                 nunits_enc=(50,20), nunits_dec=(50,20), \
                 nunits_link=(25,10), add_zero_los_frac=0.25,out_var_min=1e-4,\
                 init_bias_stddev=10.0, init_kernel_stddev=10.0,\
                 model_dir='model_data'):
        """
        Constructor

        Parameters
        ----------
        npaths_max : int
            max number of paths per link
        pl_max : scalar
            max path loss in dB
        nunits_enc : list of integers
            number of hidden units in each layer of the encoder
        nunits_dec : list of integers
            number of hidden units in each layer of the decoder
        nunits_link:  list of integers
            number of hidden units in each layer of the link classifier
        nlatent : int
            number of latent states in the VAE model
        nunits_enc : list of integers
            number of hidden units in each layer of the encoder
        nunits_dec : list of integers
            number of hidden units in each layer of the decoder  
        add_zero_los_frac: scalar
            in the link state modeling, a fraction of points at the origin
            are added to ensure the model predicts a LOS link there.
        out_var_min:  scalar
            minimum output variance.  This is used for improved conditioning 
        init_kernel_stddev : scalar
            std deviation of the kernel in the initialization
        init_bias_stddev : scalar
            std deviation of the bias in the initialization
        model_dir : string
            path to the directory for all the model files.
            if this path does not exist, it will be created             
        """
        self.npaths_max = npaths_max
        self.pl_max = pl_max
        self.ndim = 3  # number of spatial dimensions
        self.nunits_link = nunits_link
        self.init_kernel_stddev = init_kernel_stddev
        self.init_bias_stddev = init_bias_stddev
        self.model_dir = model_dir
        
        self.nlatent = nlatent
        self.nunits_enc = nunits_enc
        self.nunits_dec = nunits_dec
        self.add_zero_los_frac = add_zero_los_frac
        self.out_var_min = out_var_min        
        
        
    def get_link_state(self, los_exists, nlos_pl):
        """
        Computes the link state

        Parameters
        ----------
        los_exists : (nlink,) array of boolean
            indicates if each link has an LOS path or not
        nlos_pl : (nlink,npaths_max) array of floats
            path loss for each path in the link

        Returns
        -------
        link_state : (nlink,) array of int
            indicates link state: no_link, los_link, nlos_link            
        """
        
        # Compute number of paths for each link
        npath = np.sum((nlos_pl < self.pl_max-0.1), axis=1)
        
        # Compute link state
        Ilos  = (los_exists==1)
        Ino   = (los_exists==0) & (npath==0)
        Inlos = (los_exists==0) & (npath>0)
        
        link_state = self.los_link*Ilos + self.nlos_link*Inlos\
            + self.no_link*Ino
        return link_state
    
    def transform_link(self,dvec,cell_type,fit=False):
        """
        Pre-processes input for the link classifier network

        Parameters
        ----------
        dvec : (nlink,3) array
            vector from cell to UAV
        cell_type : (nlink,) array of ints
            cell type.  One of terr_cell, aerial_cell

        Returns
        -------
        X:  (nlink,nin_link) array:
            transformed data for input to the NN
        """
        
        # 3D distance and vertical distance.
        # Note that vertical distance can be negative
        #dx = np.sqrt(np.sum(dvec[:,0]**2, axis=1))
        dx = np.sqrt(dvec[:,0]**2 + dvec[:,1]**2)
        dz = dvec[:,2]
        
        X0 = np.column_stack((dx, dz, dx*cell_type, dz*cell_type, cell_type))
                    
        
        # Transform the data with the scaler.
        # If fit is set, the transform is also learned
        if fit:
            self.link_scaler = sklearn.preprocessing.StandardScaler()
            X = self.link_scaler.fit_transform(X0)
        else:
            X = self.link_scaler.transform(X0)
        return X
        
        
    def build_link_mod(self):
        """
        Builds the link classifier neural network            
        """              
        
        # Input layer
        self.link_mod = tfkm.Sequential()
        self.link_mod.add(tfkl.Input(self.nin_link, name='Input'))
        
        # Hidden layers
        for i, nh in enumerate(self.nunits_link):
            self.link_mod.add(tfkl.Dense(nh, activation='sigmoid', name='FC%d' % i))
        
        # Output softmax for classification
        self.link_mod.add(tfkl.Dense(self.nlink_states, activation='softmax', name='Output'))
              
    
    def add_los_zero(self,dvec,cell_type,ls):
        """
        Appends points at dvec=0 with LOS.  This is used to 
        ensure the model predicts a LOS link at zero distance.

        Parameters
        ----------
        dvec : (nlink,ndim) array
            vector from cell to UAV
        cell_type : (nlink,) array of ints
            cell type. 
        ls : (nlink,) array of ints
            link types

        Returns
        -------
        dvec, cell_type, ls : as above
            Values with the zeros appended at the end

        """
        
        ns = dvec.shape[0]
        nadd = int(ns*self.add_zero_los_frac)
        if nadd <= 0:
            return dvec, cell_type, ls
        
        I = np.random.randint(ns,size=(nadd,))
        
        # Variables to append
        cell_type1 = cell_type[I]
        ls1 = np.tile(ChanMod.los_link, nadd)
        dvec1 = np.zeros((nadd,3))
        
        # Add the points
        cell_type = np.hstack((cell_type, cell_type1))
        ls = np.hstack((ls, ls1))
        dvec = np.vstack((dvec, dvec1))
        return dvec, cell_type, ls
        
        
        
    
    def fit_link_mod(self, train_data, test_data, epochs=50, lr=1e-4):
        """
        Trains the link classifier model

        Parameters
        ----------
        train_data : dictionary
            training data dictionary.
        test_data : dictionary
            test data dictionary.    
        """      
        
        
        # Get the link state
        ytr = self.get_link_state(train_data['los_exists'], train_data['nlos_pl'])
        yts = self.get_link_state(test_data['los_exists'], test_data['nlos_pl'])
        
        
        # Get the position and cell types
        dvectr = train_data['dvec']
        celltr = train_data['cell_type']
        dvects = test_data['dvec']
        cellts = test_data['cell_type']
        
        # Fit the transforms
        self.transform_link(dvectr,celltr, fit=True)

        # Append the zero points        
        dvectr, celltr, ytr = self.add_los_zero(dvectr,celltr,ytr)
        dvects, cellts, yts = self.add_los_zero(dvects,cellts,yts)
                        
        # Transform the input to the neural network
        Xtr = self.transform_link(dvectr,celltr)
        Xts = self.transform_link(dvects,cellts)
                    
        # Fit the neural network
        opt = Adam(lr=lr)
        self.link_mod.compile(opt,loss='sparse_categorical_crossentropy',\
                metrics=['accuracy'])
        
        self.link_hist = self.link_mod.fit(\
                Xtr,ytr, batch_size=100, epochs=epochs, validation_data=(Xts,yts) )            
            
    
    def link_predict(self,dvec,cell_type):
        """
        Predicts the link state

        Parameters
        ----------
        dvec : (nlink,ndim) array
            vector from cell to UAV
        cell_type : (nlink,) array of ints
            cell type.  0 = terrestrial, 1=aerial

        Returns
        -------
        prob:  (nlink,nlink_states) array:
            probabilities of each link state

        """
        X = self.transform_link(dvec, cell_type)
        prob = self.link_mod.predict(X)
        return prob
    
    def save_link_model(self, weights_fn='link_weights.h5', preproc_fn='link_preproc.p'):
        """
        Saves link state predictor model data to files

        Parameters
        ----------
        weights_fn : string
            Filename for the link neural network weights.  This is an H5 file
        preproc_fn : string
            Filename for the pickle copy of the pre-processors

        """
        # Create the file paths
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        preproc_path = os.path.join(self.model_dir, preproc_fn)
        weigths_path = os.path.join(self.model_dir, weights_fn)
        
        
        # Save the pre-processors
        with open(preproc_path,'wb') as fp:
            pickle.dump([self.link_scaler, self.nunits_link], fp)
            
        # Save the VAE weights
        self.link_mod.save_weights(weigths_path, save_format='h5')
        
    def load_link_model(self, weights_fn='link_weights.h5', preproc_fn='link_preproc.p'):
        """
        Load link state predictor model data from files

        Parameters
        ----------
        weights_fn : string
            Filename for the VAE weights.  This is an H5 file
        preproc_fn : string
            Filename for the pickle copy of the pre-processors

        """
        # Create the file paths
        preproc_path = os.path.join(self.model_dir, preproc_fn)
        weigths_path = os.path.join(self.model_dir, weights_fn)

        # Load the pre-processors and model config
        with open(preproc_path,'rb') as fp:
            self.link_scaler, self.nunits_link = pickle.load(fp)
            
        # Build the link state predictor
        self.build_link_mod()
        
        # Load the VAE weights
        self.link_mod.load_weights(weigths_path)
        
    def build_path_mod(self):
        """
        Builds the VAE for the NLOS paths
        """
        ndat = self.npaths_max
        
        self.path_mod = CondVAE(\
            nlatent=self.nlatent, ndat=ndat, ncond=ChanMod.ncond,\
            nunits_enc=self.nunits_enc, nunits_dec=self.nunits_dec,\
            out_var_min=self.out_var_min, sort_out=True,\
            init_bias_stddev=self.init_bias_stddev,\
            init_kernel_stddev=self.init_kernel_stddev)
            
    def fit_path_mod(self, train_data, test_data, epochs=50, lr=1e-3,\
                     checkpoint_period = 0):
        """
        Trains the path model

        Parameters
        ----------
        train_data : dictionary
            training data dictionary.
        test_data : dictionary
            test data dictionary. 
        epochs: int
            number of training epochs
        lr: scalar
            learning rate
        checkpoint_period:  int
            period in epochs for saving the model checkpoints.  
            A value of 0 indicates that checkpoints are not be saved.
        """      
        # Get the link state
        ls_tr = self.get_link_state(train_data['los_exists'], train_data['nlos_pl'])
        ls_ts = self.get_link_state(test_data['los_exists'], test_data['nlos_pl'])
        los_tr = ls_tr == ChanMod.los_link
        los_ts = ls_tr == ChanMod.los_link
        
        
        # Extract the links that are in LOS or NLOS
        Itr = np.where(ls_tr != ChanMod.no_link)[0]
        Its = np.where(ls_ts != ChanMod.no_link)[0]
        
        # Fit and transform the condition data
        Utr = self.transform_cond(\
            train_data['dvec'][Itr], train_data['cell_type'][Itr],\
            los_tr[Itr], fit=True)
        Uts = self.transform_cond(\
            test_data['dvec'][Its], test_data['cell_type'][Its],\
            los_ts[Its])            
        
        # Fit and transform the data
        Xtr = self.transform_data(train_data['nlos_pl'][Itr], fit=True)
        Xts = self.transform_data(test_data['nlos_pl'][Its])
        
        # Create the checkpoint callback
        batch_size = 100
        if (checkpoint_period > 0):            
            save_freq = checkpoint_period*int(np.ceil(Xtr.shape[0]/batch_size))
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            cp_path = os.path.join(self.model_dir, 'path_weights.{epoch:03d}.h5')
            callbacks = [tf.keras.callbacks.ModelCheckpoint(\
                filepath=cp_path, save_weights_only=True,save_freq=save_freq)]
        else:
            callbacks = []
        
        
        # Fit the model
        opt = Adam(lr=lr)
        self.path_mod.vae.compile(opt)
            
        self.path_hist = self.path_mod.vae.fit(\
                    [Xtr,Utr], batch_size=batch_size, epochs=epochs,\
                    validation_data=([Xts,Uts],None),\
                    callbacks=callbacks)
        
        # Save the history
        hist_path = os.path.join(self.model_dir, 'path_train_hist.p')        
        with open(hist_path,'wb') as fp:
            pickle.dump(self.path_hist.history, fp)
        
        
    def transform_cond(self, dvec, cell_type, los, fit=False):
        """
        Pre-processing transform on the condition

        Parameters
        ----------
        dvec : (nlink,ndim) array
            vector from cell to UAV
        cell_type : (nlink,) array of ints
            cell type.  One of terr_cell, aerial_cell
        los:  (nlink,) array of booleans
            indicates if link is in LOS or not
        fit : boolean
            flag indicating if the transform should be fit

        Returns
        -------
        U : (nlink,ncond) array
            Transform conditioned features
        """
      
        
        # 3D distance and vertical distance.
        # Note that vertical distance can be negative
        d3d = np.sqrt(np.sum(dvec**2, axis=1))
        dvert = dvec[:,2]
        
        # Transform the condition variables
        U0 = np.column_stack((d3d, np.log10(d3d), dvert, cell_type, los))
        self.ncond = U0.shape[1]
        
        # Transform the data with the scaler.
        # If fit is set, the transform is also learned
        if fit:
            self.cond_scaler = sklearn.preprocessing.StandardScaler()
            U = self.cond_scaler.fit_transform(U0)
        else:
            U = self.cond_scaler.transform(U0)
            
        return U
      
        
    def transform_data(self, pl, fit=False):
        """
        Fits the pre-processing transform on the data

        Parameters
        ----------
        pl : (nlink,npaths_max) array 
            path losses of each path in each link.
            A value of pl_max indicates no path
        fit : boolean
            flag indicating if the transform should be fit            

        Returns
        -------
        X : (nlink,ndat) array
            Transform data features
        """
        
        # Transform the condition variables
        X0 = self.pl_max - pl[:,:self.npaths_max]     
        
        # Transform the data with the scaler.
        # If fit is set, the transform is also learned
        if fit:
            self.dat_scaler = sklearn.preprocessing.MinMaxScaler()
            X = self.dat_scaler.fit_transform(X0)
        else:
            X = self.dat_scaler.transform(X0)
        return X
    
    def inv_transform_data(self, X):
        """
        Inverts the pre-processing transform on the data

        Parameters
        ----------
        X : (nlink,ndat) array 
            Pre-processed data

        Returns
        -------
        pl : (nlink,npaths_max) array 
            path losses of each path in each link.
            A value of pl_max indicates no path
        """
        
        # Invert the scaler
        X = np.maximum(0,X)
        X = np.minimum(1,X)
        X0 = self.dat_scaler.inverse_transform(X)
        
        # Sort and make positive
        X0 = np.maximum(0, X0)
        X0 = np.fliplr(np.sort(X0, axis=-1))
        
        # Transform the condition variables
        pl = self.pl_max - X0  
                
        return pl
        
    
    def sample_path(self, dvec, cell_type, los):
        """
        Generates random samples of the path data using the trained model

        Parameters
        ----------
        dvec : (nlink,ndim) array
            vector from cell to UAV
        cell_type : (nlink,) array of ints
            cell type.  One of terr_cell, aerial_cell
        los:  (nlink,) array of booleans
            indicates if link is in LOS or not
   
        Returns
        -------
        pl : (nlink,npaths_max) array 
            path losses of each path in each link.
            A value of pl_max indicates no path

        """
        
        # Get the condition variables and random noise
        U = self.transform_cond(dvec, cell_type, los)
        nlink = U.shape[0]
        Z = np.random.normal(0,1,(nlink,self.nlatent))
        
        # Run through the sampling network
        X = self.path_mod.sampler([Z,U]) 
        
        # Compute the inverse transform to get back the path loss
        pl = self.inv_transform_data(X)
        return pl
    
    def save_path_model(self, weights_fn='path_weights.h5', preproc_fn='path_preproc.p'):
        """
        Saves model data to files

        Parameters
        ----------
        weights_fn : string
            Filename for the VAE weights.  This is an H5 file
        preproc_fn : string
            Filename for the pickle copy of the pre-processors

        """
        # Create the file paths
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        preproc_path = os.path.join(self.model_dir, preproc_fn)
        weigths_path = os.path.join(self.model_dir, weights_fn)
        
        # Save the pre-processors
        with open(preproc_path,'wb') as fp:
            pickle.dump([self.dat_scaler, self.cond_scaler, self.pl_max,\
                         self.npaths_max, self.nlatent, self.nunits_enc,\
                         self.nunits_dec], fp)
            
        # Save the VAE weights
        self.path_mod.vae.save_weights(weigths_path, save_format='h5')
        
    def load_path_model(self, weights_fn='path_weights.h5', preproc_fn='path_preproc.p'):
        """
        Load model data from files

        Parameters
        ----------
        weights_fn : string
            Filename for the VAE weights.  This is an H5 file
        preproc_fn : string
            Filename for the pickle copy of the pre-processors

        """
        # Create the file paths
        preproc_path = os.path.join(self.model_dir, preproc_fn)
        weights_path = os.path.join(self.model_dir, weights_fn)
        
        # Load the pre-processors
        with open(preproc_path,'rb') as fp:
            self.dat_scaler, self.cond_scaler, self.pl_max,\
                self.npaths_max, self.nlatent, self.nunits_enc,\
                self.nunits_dec = pickle.load(fp)
            
        # Build the path model
        self.build_path_mod()
            
        # Load the VAE weights
        self.path_mod.vae.load_weights(weights_path)
        
def set_initialization(mod, layer_names, kernel_stddev=1.0, bias_stddev=1.0):
    """
    Sets the bias and kernel initializations for a set of dense layers

    Parameters
    ----------
    mod:  Tensorflow model
        Model for which the initialization is to be applied
    layer_names : list of strings
        List of names of layers to apply the initialization
    kernel_stddev : scalar
        std deviation of the kernel in the initialization
    bias_stddev : scalar
        std deviation of the bias in the initialization            
    """
    for name in layer_names:
        layer = mod.get_layer(name)
        nin = layer.input_shape[-1]
        nout = layer.output_shape[-1]
        W = np.random.normal(0,kernel_stddev/np.sqrt(nin),\
                             (nin,nout)).astype(np.float32)
        b = np.random.normal(0,bias_stddev,\
                             (nout,)).astype(np.float32)
        layer.set_weights([W,b])
    