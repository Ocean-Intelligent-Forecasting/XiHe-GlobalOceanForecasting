import os
import onnxruntime
import numpy as np
import argparse
import pathlib
import xarray as xr
import pandas as pd
from data_process import process_data
import torch

file_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(file_path)

class inference_onnx():
    def __init__(self, onnx_file_path, yaml_path, origin_input, input, output):
        super().__init__()
        self.onnx_file_path = onnx_file_path
        self.yaml_path = yaml_path
        self.data_processing = process_data(yaml_path, origin_input, input, output)
        self.out_transforms = self.data_processing.get_denormalize()

    def inference(self, x):
        x = self.data_processing.read_data(x)
        ort_inputs = {'input': x}
        # Use CPU as default
        providers = ['CPUExecutionProvider']
        # Use GPU if available
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        ort_session = onnxruntime.InferenceSession(self.onnx_file_path, providers=providers)
        ort_output = ort_session.run(None, ort_inputs)[0]
        output = self.out_transforms(torch.from_numpy(ort_output))
        return output

def npy2nc(npy_data):
    """
    Convert numpy format into nc file
    """
    # latitude & longitude
    lat_path = os.path.join(file_path, 'mercator_lat.npy')
    lon_path = os.path.join(file_path, 'mercator_lon.npy')
    lat, lon = np.load(lat_path), np.load(lon_path)

    # depth
    depth = np.array(
        [0.4940, 2.6457, 5.0782, 7.9296, 11.4050, 15.8101, 21.5988, 29.4447, 40.3441, 55.7643, 77.8539, 92.3261, 109.7293,
         130.6660, 155.8507, 186.1256, 222.4752, 266.0403, 318.1274, 380.2130, 453.9377, 541.0889, 643.5668])

    variables = ['thetao', 'so', 'uo', 'vo', 'zos', 'sst']
    layer_index = [
                       [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90],
                       [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91],
                       [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92],
                       [5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93],
                       0,
                       1
                  ]
    for i in range(len(variables)):
        nc_file_path = os.path.join(args.save_path, datatime_str+ "_" + variables[i] + '.nc')
        coords_dict = {
            'time': pd.date_range(datatime_str[0:4] + "-" + datatime_str[4:6] + "-" + datatime_str[6:] + " 00:00:00", periods=1),
            'latitude':lat.astype(np.float32),
            'longitude':lon.astype(np.float32)
        }
        dims = ['time','latitude','longitude']
        if i < 4:
            dims = ['time', 'depth','latitude','longitude']
            coords_dict['depth']=depth.astype(np.float32)

        nc_dataset = xr.DataArray(npy_data[:,layer_index[i], :, :].astype(np.float32), dims=dims, coords=coords_dict).to_dataset(name=variables[i])
        nc_dataset.to_netcdf(nc_file_path)    
            
if __name__ == "__main__":
    input_path = os.path.join(project_path,'input_data')
    output_path = os.path.join(project_path,'output_data')
    parser = argparse.ArgumentParser(description='inference program of xihe')
    parser.add_argument('--input_path', type=str, default=input_path, help='input data path:{0}'.format(input_path))
    parser.add_argument('--lead_day', type=int, default=7, help='lead time')
    parser.add_argument('--save_path', type=str, default=output_path, help='output data path:{0}'.format(output_path))
    args = parser.parse_args()

    # date of the input
    datatime_str = '20190101'
    depth_list = [
                    {
                        # suface input file path
                        'input_path': os.path.join(args.input_path, 'input_surface_data','input_surface_{0}.npy'.format(datatime_str)),
                        "output_data":None,
                        'mask_path': os.path.join(file_path, 'mask_surface.npy'),  
                        'layer': '1to22'
                    },
                    {
                        # deep input file path
                        'input_path': os.path.join(args.input_path, 'input_deep_data','input_deep_{0}.npy'.format(datatime_str)),
                        "output_data":None,
                        'mask_path': os.path.join(file_path, 'mask_deep.npy'), 
                        'layer': '23to33'
                    }
                 ]
    yaml_path = os.path.join(file_path, 'data.yaml')

    for depth in depth_list:
        input_path = depth['input_path']
        print(input_path)
        layer = depth['layer']
        # Load the input numpy arrays
        x = np.load(input_path).astype(np.float32)
        x = torch.tensor(x)
        # file path of the trained model
        onnx_path = os.path.join(project_path, 'models/xihe_{0}_{1}day.onnx'.format(layer,str(args.lead_day)))
        y = inference_onnx(onnx_path, yaml_path, f'variables_{layer}',f'input_{layer}', f'output_{layer}').inference(x)
        mask_file = np.load(depth['mask_path'])
        y[mask_file] = np.nan
        depth['output_data'] = y
	# Merge the results of two numpy variables into one numpy variable
    npy_data = np.concatenate([depth_list[0]['output_data'], depth_list[1]['output_data']], axis=1)
    # Convert numpy format into nc file
    npy2nc(npy_data)

