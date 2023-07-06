# Coralshift: Predicting present reef cover from historic environmental conditions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Coral reefs are biodiversity hotspots that play a pivotal role in the health of global marine ecosystems, as well as protecting coastlines and sustaining coastal communities throughout the tropics. Coral reefs are under intense and increasing stress from global climate change and poor land management practices, giving rise to temperature increases, ocean acidification, sea-level rise, and coastal pollution. Together, these stressors pose an existential risk to coral reefs, which are threatened with functional extinction in many parts of the world over the coming decades.

There has been considerable innovation and progress in coral reef conservation, but protecting coral reefs in the long term requires acknowledging and working within the context of global climate change. The geographic distribution of regions able to support coral reefs will shift under climate change, and significant areas of current reef will not survive. It is therefore essential to focus conservation efforts on regions where reefs might survive in future, and on regions that may become suitable in the coming decades. To guide this conservation effort, we need tools that can robustly predict the presence and health of coral reefs from the relatively coarse-scale data available in climate models.

This three-month research project will aim to answer the how, when, and why this migration will occur by creating a predictive model of reef presence using oceanographic, land use, and coastal topographic data.

### Contributing
This project forms an assessed component of the Master of Research certificate as part of the Artificial Intelligence for the study of Environmental Risks ([AI4ER]((https://ai4er-cdt.esc.cam.ac.uk/)) CDT at the University of Cambridge. There is therefore no direct opportunity for collaboration. If this research interests you – if you have any questions, suggestions, or comments – please do get in touch. My website and contacts can be accessed [here](https://orlando-code.github.io/). I am always happy to learn from others, or hear about potential future collaborations.

### License
This repository and associated code is governed by an MIT license. Please see the LICENSE.md file in the main repository folder for more information.

---

# Using this repository
You're encouraged to explore the code in the accompanying notebooks. These allow you to download and visualise the necessary data, train and test machine learning models.

## Environment setup
It is recommended to use a package manager such as `conda` to install the dependencies. 
You may install with `pip`. However, since the package utilises the `rasterio` module – which requires C compiled code – successful installation via `pip` is not guaranteed for Windows machines.

### CONDA | CUDA
```shell
conda env create -f environment.yml
```

### PIP
You may want to install it in a virtual environment.
```shell
pip install -r requirements.txt
```

## Loading the datasets

### Example datasets
'Toy' datasets – simplified recreations of the actual data used – are hosted on [Zenodo](10.5281/zenodo.8110925) (DOI: 10.5281/zenodo.8110925). These allow rapid recreation of key results.

### Full datasets
Datasets used in the accompanying report are downloaded via the `notebooks/dataloading.ipynb` notebook. This details the necessary steps to download and process the data. This functionality uses a Jupyter notebook to enable subsets of the data to be easily installed, and to make data installation accessible to all users through the notebook's visual format. Note: due to the sequential method by which APIs are queried, this download takes on the order of days to complete and requires a large amount of storage space. It is highly recommended that the example datasets (above) are used.

Given the size of data ~10s GB, it is recommended to use a remote storage service, such as [Google Drive](https://www.google.co.uk/intl/en-GB/drive/).

After downloading, the data will be organised in the following directory structure. Capitalised words indicate where the user has a choice in specifying the data, e.g. choosing the resolution or the region of interest.

`/datasets/` (situated in parent folder of `coralshift` repository)

```
 ├── bathymetry  
 │      ├── REGION_NAME_1_2020_30m_MSL_cog.tif  
 │      ├── REGION_NAME_2_2020_30m_MSL_cog.tif
 │      │── ...  
 │      ├── REGION_NAME_N_2020_30m_MSL_cog.tif 
 │      ├── bathymetry_REGION_NAME_1_RESOLUTION.nc
 │      ├── bathymetry_REGION_NAME_2_RESOLUTION.nc
 │      │── ...  
 │      └── bathymetry_REGION_NAME_N_RESOLUTION.nc
 |
 ├── gradients
 │      ├── region_REGION_NAME_1_RESOLUTION_gradients.nc
 │      ├── region_REGION_NAME_2_RESOLUTION_gradients.nc
 │      │── ...  
 │      └── region_REGION_NAME_N_RESOLUTION_gradients.nc
 |
 |
 ├── reef_baseline  
 │      ├── region_name_N  
 │      │      └── benthic.gpkg
 │      │      └── benthic.pkl
 │      │      └── REGION_NAME shapefile components
 │      └── gt_files  
 │      │      └── RESOLUTION_arrays
 │      │            |── coral_region_REGION_NAME_1_1000m_RESOLUTION.nc
 │      │            |── coral_region_REGION_NAME_2_1000m_RESOLUTION.nc
 │      │            |── ...
 │      │            └── coral_region_REGION_NAME_N_1000m_RESOLUTION.nc
 │      │── coral_region_REGION_NAME_1_1000m.tif
 │      │── coral_region_REGION_NAME_2_1000m.tif
 │      │── ...
 │      └── coral_region_REGION_NAME_N_1000m.tif
 |
 |── global_ocean_reanalysis  
 │      ├── daily_means
 │      │      └── region_name e.g. Great_Barrier_Reef_A
 │      │            |── var1 (.nc files and accompanying metadata containing variable data grouped by year)
 │      │            |── var2
 │      │            |── ...
 │      │            |── varN
 │      │            |── merged_vars (.nc files and accompanying metadata for each variable)
 │      │            └── cmems_gopr_daily_REGION_NAME.nc (merged variables for region + metadata)
 │      └── monthly_means
 │             └── region_name e.g. Great_Barrier_Reef_A
 │                   |── var1 (.nc files and accompanying metadata containing variable data grouped by year)
 │                   |── var2
 │                   |── ...
 │                   |── varN
 │                   |── merged_vars (.nc files and accompanying metadata for each variable)
 │                   └── cmems_gopr_monthly_REGION_NAME.nc (merged variables for region + metadata)
 | 
 └── era5
        └── region_name e.g. Great_Barrier_Reef_A
               │── var1 (.nc files containing variable data grouped by year)
               │── var2
               │── ...
               │── varN
               └── weather_parameters (.nc file for each variable for whole specified time period)
```


