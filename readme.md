# UAV millimeter wave channel modeling 

The millimeter wave (mmWave) bands are being increasingly considered for wireless communication to UAVs (unmanned aerial vehicles, also called drones).  Communincation in these frequencies offers the possibility of supporting massive data rates at low latency for real-time sensor and camera data transfer, remote control, and situations when the UAV acts as an aerial base station.  This project seeks to use machine learning techniques to model the channel between the UAV and ground base stations.   The project is still under work.

## Data 
The data comes from simulations using powerful ray tracer, [RemCom Wireless Insite](https://www.remcom.com/wireless-insite-em-propagation-software).  RemCom generously donated this tool for this project.  Several UAV and cell locations were placed in an urban region and the channel characteristics between each UAV-cell were simulated.  There are two cell types considered:
* Aerial cells:  These are cell sites located on rooftops and would be targeted for aerial coverage.
* Terrestrial cells:  These are cell sites located at street level locations primarily targeting terrestrial users.  We include these sites in the simulation to determine the interference from UAVs and also to see if these cells can provide coverage, at least for low flying UAVs.

The final data consists of over 20000 links.  Each link is decribed by a set of paths with each path having a path loss, delay and angles of arrival and departure.  

## Modeling
When this project is completed, the models has two components:
*  A *link predictor model* that predicts the link state (LOS,NLOS and no link) from the link conditions.  The link conditions are the 3D TX-RX vector and the cell type and is based on a simple neural network.
* A *path model* that generates random samples of the path data (path losses, gains, and angles) from the link conditions.  This is modeled using a conditional variational auto-encoder.

## Running the programs
The main files are:
* `create_data.py`:  This creates the training and test data from the Remcom data.  You do not need to re-run this program since the training and test data `train_test_data.p` is included in the repository.
* `train_mod.py`:  This will train the link and path models.  You can run this with the command:
   ```
   python3 train_mod.py --nepochs_path 2000 --model_dir model_data
   ```
   Right now, you have to run the training for a huge nmber of epochs for decent results.  Not sure why. If you would like to speed things up, run this on the NYU HPC cluster by witht the [following commands](./hpc_notes.md).
* `plot_los_prob.py`:  This will produce a plot of the predicted LOS probabilites as a function of distance.
* `plot_path_loss_cdf.py`:  This will produce a plot of the path loss CDF predicted by the model and compares the CDF on the test data.  
   
