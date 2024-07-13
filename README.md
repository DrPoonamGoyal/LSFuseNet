LSFuseNet fuses histogram time series from two satellites - Landsat-8 and Sentinel-2. Can be used for multiple earth observation applications

Additionally we can include meteorological and soil data (if applicable to application)

%%%%%%%%% Data Download %%%%%%%%%%%%%%%%%%

Download histograms individual csv files from GEE. 

*This data needs pre-processing for missing values and preparation in the form suitable to be used in ML models i.e. conversion into .npz files to be used as input to ML/DL models


%%%%%%%%% Data Preparation %%%%%%%%%%%%%%%%%%

Data preparation process is done individually for each satellite, each crop. Convert histogram csvs into .npz files for each location-year pair separately for each satellite.

In addition to this we need to prepare meteorological data as per the temporal resolution of each satellite. 


%%%%%%%%% Model-LSFuseNet %%%%%%%%%%%%%%%%%%

The kernel size of CNN module in encoders need to be changes as per satellite and the timeseries length in LSTM is different depending on satellite and crop.
