'''
1、install OnnxRun-time gpu library and yaml library, if yaml library does not exist, install pyyaml library
2、Modify the onnx path, yaml path, and mean-SD file path
3、operation
'''


from datetime import datetime, timedelta
import onnxruntime
import numpy as np
import os
import argparse
import pathlib
import torchdata.datapipes as dp
from torchvision.transforms import transforms
from data_process_32 import process_data
import torch

file_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(file_path)
input_path = os.path.join(project_path, 'examples', 'input_data')
output_path = os.path.join(project_path, 'examples', 'output_data')

class inference_cpu():
    def __init__(self, onnx_file_path, yaml_path, origin_input, input, output):
        super().__init__()
        self.onnx_file_path = onnx_file_path
        self.yaml_path = yaml_path
        self.data_processing = process_data(yaml_path, origin_input, input, output)
        self.out_transforms = self.data_processing.get_denormalize()

    def inference(self, x):
        x = self.data_processing.read_data(x)  # Standardize the data
        ort_inputs = {'input': x}
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        ort_session = onnxruntime.InferenceSession(self.onnx_file_path, providers=providers)
        ort_output = ort_session.run(None, ort_inputs)[0]
        output = self.out_transforms(torch.from_numpy(ort_output))
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reasoning program')
    parser.add_argument("--input_path", type=str, default=input_path, help="Enter a data path, for example: {0}".format(input_path))
    parser.add_argument("--lead_day", type=int, default=7, help="The forecast lead time, in days, is a number ranging from 1 to 10. The default value is 7")
    parser.add_argument("--save_path", type=str, default=output_path, help="Output forecast data path, for example:{0}".format(output_path))
    args = parser.parse_args()

    input_surface_path = os.path.join(args.input_path, 'input_surface_data')
    input_deep_path = os.path.join(args.input_path, 'input_deep_data')
    output_surface_path = os.path.join(args.save_path, 'output_surface_data')
    output_deep_path = os.path.join(args.save_path, 'output_deep_data')
    mask_surface_path = os.path.join(file_path, 'mask_surface.npy')
    mask_deep_path = os.path.join(file_path, 'mask_deep.npy')
    pathlib.Path(output_surface_path).mkdir(exist_ok=True, parents=True)
    pathlib.Path(output_deep_path).mkdir(exist_ok=True, parents=True)

    if not (args.lead_day >= 1 and args.lead_day <= 10):
        raise ValueError("The lead day parameter must be a number ranging from 1 to 10")

    depth_list = [{"input_path": input_surface_path, "output_path": output_surface_path, "mask_path": mask_surface_path, "layer": "1to22"},
        {"input_path": input_deep_path, "output_path": output_deep_path, "mask_path": mask_deep_path, "layer": "23to33"}]

    for depth in depth_list:
        data_path_list = list(dp.iter.FileLister(depth.get("input_path")))
		
        mask_file = np.load(depth.get("mask_path"))
        
        onnx_path = os.path.join(project_path, "models/xihe_{0}_{1}day.onnx".format(depth.get("layer"),str(args.lead_day))) # onnx path
        yaml_path = os.path.join(file_path, 'data.yaml')  # yaml file path
        print('loading model from:', onnx_path)
        for path in data_path_list:
            x = np.load(path).astype(np.float32)
            file_name = path.split('/')[-1]
            date_string = file_name.split('_')[1].split('.')[0]  # date
            print("cur_date:", date_string)
            date = datetime.strptime(date_string, "%Y%m%d")
            save_date = date + timedelta(days=args.lead_day)
            pred_date = save_date.strftime("%Y%m%d")  # creation date
            print("pred_date:", pred_date)
            x = torch.tensor(x)
            y = inference_cpu(onnx_path, yaml_path, f'variables_{depth.get("layer")}',
                              f'input_{depth.get("layer")}', f'output_{depth.get("layer")}').inference(x)
            y = y.cpu().numpy()
            y[mask_file] = np.nan
            np.save(os.path.join(depth.get("output_path"), "pred_mra5_" + pred_date + '.npy'), y)



