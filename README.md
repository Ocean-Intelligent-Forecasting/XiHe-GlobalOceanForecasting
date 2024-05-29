# XiHe-GlobalOceanForecasting

This is the official repository for the Xihe papers.

[XiHe: A Data-Driven Model for Global Ocean Eddy-Resolving Forecasting](https://arxiv.org/abs/2402.02995), arXiv preprint arXiv:2402.02995, 2024.

*by Xiang Wang, Renzhi Wang, Ningzi Hu, Pinqiang Wang, Peng Huo, Guihua Wang, Huizan Wang, Senzhang Wang, Junxing Zhu, Jianbo Xu, Jun Yin, Senliang Bao, Ciqiang Luo, Ziqing Zu, Yi Han, Weimin Zhang, Kaijun Ren, Kefeng Deng, Junqiang Song* 

Resources including pre-trained models, and inference code are released here.



## Installation

The downloaded files shall be organized as the following hierarchy:

```plain
├── root
│   ├── input_data
│   │   ├── input_surface_data
│   │   │	├── input_surface_20190101.npy
│   │   ├── input_deep_data
│   │   │	├── input_deep_20190101.npy
│   ├── output_data
│   ├── models
│   |   ├── xihe_1to22_1day.onnx
│   |   ├── ...
│   |   ├── xihe_1to22_10day.onnx
│   |   ├── xihe_23to33_1day.onnx
│   |   ├── ...
│   |   ├── xihe_23to33_10day.onnx
│   ├── src
│   |   ├── data.yaml
│   |   ├── normalize_mean_50.npz
│   |   ├── normalize_std_50.npz
│   |   ├── mask_surface.npy
│   |   ├── mask_deep.npy
│   |   ├── mercator_lat.npy
│   |   ├── mercator_lon.npy
│   |   ├── data_process.py
│   |   ├── inference.py
```

First install the packaged virtual environment `pycdo`([Baidu netdisk]([https://pan.baidu.com/s/18K46vXC7qFcnABHbuR-r-A?pwd=ubnx](https://pan.baidu.com/s/1Lth8ZSlo-kuOif37jNcI-g?pwd=jceh)) or [Google netdisk](https://drive.google.com/drive/folders/1eZyNOJUTwFVdG19qEpGQ6ZsZg9Nu-Y4L?usp=drive_link) and activate the virtual environment.



## Global ocean forecasting (inference) using the trained models

#### Downloading trained models

Please download the layers 1to22 and 23to33 pre-trained models for 1 to 10 days from [Baidu netdisk]([https://pan.baidu.com/s/18K46vXC7qFcnABHbuR-r-A?pwd=ubnx](https://pan.baidu.com/s/1Lth8ZSlo-kuOif37jNcI-g?pwd=jceh) or [Google netdisk](https://drive.google.com/drive/folders/1eZyNOJUTwFVdG19qEpGQ6ZsZg9Nu-Y4L?usp=drive_link).

These models are stored using the ONNX format, and thus can be used via different languages such as Python, C++, C#, Java, etc.

#### Input data 

We support [GLORYS12 reanalysis](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description) as initial fields, please prepare the input data using [numpy](https://numpy.org/) and transformer it into a `.npy` file using the netCDF4 package.  

There are two input data files that shall be put under the `input_data/input_surface_data` and `input_data/input_deep_data`, which stores the input data for 1 to 22 layers and  23 to 33  layers respectively. The specific details of layers and variables can be found in the following text. 

We provide an example of preprocessed input data, `input_surface_20190101.npy` and `input_deep_20190101.npy`, which correspond to the [daily means of GLORYS12 reanalysis](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/files?subdataset=cmems_mod_glo_phy_my_0.083deg_P1D-m_202311) data at 2019/01/01. Please download them from [Baidu netdisk]([https://pan.baidu.com/s/18K46vXC7qFcnABHbuR-r-A?pwd=ubnx](https://pan.baidu.com/s/1Lth8ZSlo-kuOif37jNcI-g?pwd=jceh)) or [Google netdisk](https://drive.google.com/drive/folders/1eZyNOJUTwFVdG19qEpGQ6ZsZg9Nu-Y4L?usp=drive_link).

#### Inference 

Running the following command, one can get the 7-day ocean forecast in the `output_data` folder. 

```
python inference.py --lead_day 7 --save_path output_data
```



## Data description

### Data sources

We use three data to train Xihe:

- [**GHRSST**](https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001/files?subdataset=METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2&path=SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001%2FMETOFFICE-GLO-SST-L4-NRT-OBS-SST-V2%2F): satellite **SST** data from the Operational Sea Surface Temperature and Ice Analysis (OSTIA) .
- [**ERA5**](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form): 10m wind field of **u, v**.
- [**GLORYS12 Reanalysis**](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/files?subdataset=cmems_mod_glo_phy_my_0.083deg_P1D-m_202311) : including **ocean temperature**, **salinity**, **the zonal and meridional components of ocean currents** with a total of 23 layers (i.e.: the 1st layer: 0.49m, the 3rd layer: 2.65m, the 5th layer: 5.08m, the 7th layer: 7.93m, the 9th layer: 11.41m, the 11th layer: 15.81m, the 13th layer: 21.60m, the 15th layer: 29.44m, the 17th layer: GLORYS, the 19th layer: 55.76m, the 21st layer: 77.85m, the 22nd layer: 92.32m, the 23rd layer: 109.73m, the 24th layer: 130.67m, the 25th layer: 155.85m, the 26th layer: 186.13m, the 27th layer: 222.48m, the 28th layer: 266.04m, the 29th layer: 318.31m, the 30th layer: 380.21m, the 31st layer: 453.94m, the 32nd layer: 541.09m, and the 33rd layer: 643.57m)，and **sea surface height above geoid**.

> The main training data of Xihe is the GLORYS12 reanalysis. We also support other ocean data as initial fields, but **the actual depth of the input data layers needs to correspond to the above selected depth**. 

### Input data

- `input_surface_data` stores the input data for 1 to 22 layers. It is a NumPy array shaped **(1,52,2041,4320)** which represents the variables **(Time, Variables, Lat, Lon)**.  The  Variables are organized in the following order: **zos** (sea surface height above geoid), **u** (era5 10m zonal component of sea surface wind), **v** (era5 10m meridional component of sea surface wind), **sst** (sea surface temperature), **thetao_0, so_0, uo_0, vo_0,..., thetao_21, so_21, uo_21, vo_21** (the 1st layer of ocean temperature, the 1st layer of ocean salinity, the zonal component of the 1st layer of ocean currents, the meridional component of the 1st layer of ocean currents, ..., the 22nd layer of ocean temperature, the 22nd layer of ocean salinity, the zonal component of the 22nd layer of ocean currents, the meridional component of the 22nd layer of ocean currents).

- `input_deep_data` stores the input data for 23 to 33 layers. It is a NumPy array shaped **(1,48,2041,4320)** which represents the variables **(Time, Variables, Lat, Lon)**. The  Variables are organized in the following order: **zos** (sea surface height above geoid), **u** (era5 10m zonal component of sea surface wind), **v** (10m meridional component of sea surface wind), **sst** (sea surface temperature), **thetao_22, so_22, uo_22, vo_22,..., thetao_32, so_32, uo_32, vo_32** (the 23rd layer of ocean temperature, the 23rd layer of ocean salinity, the zonal component of the 23rd layer of ocean currents, the meridional component of the 23rd layer of ocean currents, ..., the 33rd layer of ocean temperature, the 33rd layer of ocean salinity, the zonal component of the 33rd layer of ocean currents, the meridional component of the 33rd layer of ocean currents).

> In both cases, the dimensions of 2041 and 4320 represent the size along the latitude and longitude, where the numerical range is [-80,90] degree and [-180,180] degree, respectively, and the spacing is 1/12 degrees. For each 2041x4320 slice, the data format is exactly the same as the `.nc` file download from the [GLORYS12 reanalysis](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description) official website.
>
> Note that the NumPy arrays should be in single precision (`.astype(np.float32)`).

### Output data

The model predicts 6 ocean variables. There are 23 layers including:  **ocean temperature, salinity, zonal and meridional components of ocean current** (i.e., layer 1: 0.49m, layer 3: 2.65m, layer 5: 5.08m, layer 7: 7.93m, layer 9: 11.41m, layer 11: 15.81m, layer 13: 21.60m, layer 15: 29.44m, layer 17: 40.34m, layer 19: 55.76m, layer 21: 77.85m, layer 22: 92.32m, layer 23: 109.73m, layer 23: 109.73m. The 24th layer: 130.67m, the 25th layer: 155.85m, the 26th layer: 186.13m, the 27th layer: 222.48m, the 28th layer: 266.04m, the 29th layer: 318.31m, the 30th layer: 380.21m, the 31st layer: 453.94m, the 32nd layer: 541.09m and the 33rd layer: 643.57m)，**sea surface height above geoid** and **sea surface temperature**.




## References

If you use the resource in your research, please cite our paper:
```tex
@article{wang2024xihe,
  title={XiHe: A Data-Driven Model for Global Ocean Eddy-Resolving Forecasting},
  author={Wang, Xiang and Wang, Renzhi and Hu, Ningzi and Wang, Pinqiang and Huo, Peng and Wang, Guihua and Wang, Huizan and Wang, Sengzhang and Zhu, Junxing and Xu, Jianbo and others},
  journal={arXiv preprint arXiv:2402.02995},
  year={2024}
}
```
