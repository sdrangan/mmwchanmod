# UAV Millimeter Wave Channel Modeling Using Variational Auto-Encoders

* Sundeep Rangan, William Xia, Marco Mezzavilla, Giuseppe Loianno (NYU)
* Giovanni Geraci, Angel Lozano (UPF, Barcelona)
* Vasilii Semkin (VTT, Finland)

The millimeter wave (mmWave) bands are being increasingly considered for wireless communication to UAVs (unmanned aerial vehicles, also called drones).  Communincation in these frequencies offers the possibility of supporting massive data rates at low latency for real-time sensor and camera data transfer, remote control, and situations when the UAV acts as an aerial base station. Critical to evaluating algorithms for UAVs are statistical channel models that describe the distribution of channel parameters seen in typical scenarios.  Algorithms for many procedures including beamforming and equalization require so-called *double-directional wideband models* where the channel in each link is described by a set of paths with each path having a path loss, delay and angles of arrival and departure.  Wideband double directional descriptions have large numbers of parameters with potentially complex statistical relationships.  This project seeks to use machine learning techniques to develop generative statistical models from data.

##  Generative Model
This project considers a generative model based on a cascade of two networks:
*  A *link predictor network* that predicts the link state (LOS,NLOS and no link) from the link conditions.  The link conditions are the 3D TX-RX vector and the cell type .   The cell type is either a terrestrial street-level cell or aerial rooftop-mounted cells as described below.  The link predictor network is realized using a fully connected neural network.
* A *path model* that generates random samples of the path data (path losses, delays, and angles) from the link conditions and link state.  This is modeled using a conditional variational auto-encoder (CVAE).  The CVAE encoder and decoder are also realized as fully connected neural networks.

## Raytracing Data
Training large neural networks requires large amounts of data, not currently available from actual channel measurements.  The data in this simulation thus comes from simulations using powerful ray tracer, [RemCom Wireless Insite](https://www.remcom.com/wireless-insite-em-propagation-software).  RemCom generously donated this tool for this project.  Several UAV and cell locations were placed in an urban region and the channel characteristics between each UAV-cell were simulated.  There are two cell types considered:
* Aerial cells:  These are cell sites located on rooftops and would be targeted for aerial coverage.
* Terrestrial cells:  These are cell sites located at street level locations primarily targeting terrestrial users.  We include these sites in the simulation to determine the interference from UAVs and also to see if these cells can provide coverage, at least for low flying UAVs.  

Each sample consists of one link from one UAV location to one cell location.  There are 180 UAV locations and 120 cell locations resulting in a final data set consists of 180 x 120=21600 samples.   You can directly download the training and test data with the command:
```
   python3 download.py --train_test_data
```
This will create a pickle file, `train_test_data.p`, that contains dictionaries `train_data` and `test_data` with all the data.  But, if you wish to create the data from scratch:
*  Run the command `python3 download.py --raw_data`.  This will download and extract the raw data into the directory `uav_ray_trace`.
*  Run the command `python3 create_data.py`.  This will create the file `train_test_data.p` in your local directory.

## Training the Modeling 
Once the data is downloaded, you can train the network with the command:
```
python3 train_mod.py --nepochs_path 10000 --model_dir model_data
```
The program will train both the link predictor network and the path CVAE.  The weights for both networks and other data will be stored in the directory `model_data`. 

Since the training will require a large number of epochs, you will likely want to run this on the HPC cluster.  Instructions for running on the NYU cluster can be found [here](./hpc_notes.md).  The cluster is also useful for running multiple training instances for hyper-parameter optimization.  

## Downloading a Pre-Trained Model
To avoid re-training the model from scratch, you can download the pre-trained model with `python3 download.py --model_data`. This will create the `model_data` directory in your local path.

## Using the Model
Once you have either trained the model from scratch or downloaded the pre-trained model, you can see examples of its use:
* `plot_los_prob.py`:  This will produce a plot of the predicted LOS probabilites as a function of the horizontal and vertical distance.
* `plot_path_loss_cdf.py`:  This will produce a plot of the path loss CDF predicted by the model and compares the CDF on the test data.  
* `plot_path_loss_cdf.py`:  This will produce a plot of the RMS delay spread CDF predicted by the model and compares the CDF on the test data.  
* `plot_snr.py`:  Plots the predicted median SNR in a single cell as a function of horizontal and vertical position.
   
   
## Acknowledgments
W.  Xia,  M.  Mezzavilla  and  S.  Rangan  were  supportedby  NSF  grants  1302336,  1564142,  1547332,  and  1824434,NIST, SRC, and the industrial affiliates of NYU WIRELESS.A.  Lozano  and  G.  Geraci  were  supported  by  the  ERC  grant694974,  by  MINECO’s  Project  RTI2018-101040,  and  by  the Junior Leader Program from "la Caixa" Banking Foundation.
