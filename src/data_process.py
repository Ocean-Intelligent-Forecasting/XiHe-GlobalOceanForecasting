from torchvision.transforms import transforms
import torch
import numpy as np   
import yaml
import os
file_path = os.path.dirname(os.path.abspath(__file__))

class process_data():
    def __init__(self,yaml_path,origin_input,input,output):
        super().__init__()
        self.yaml_path = yaml_path
        self.content = self.read_configs()
        self.origin_input = origin_input
        self.input = input
        self.output = output
        self.transforms = self.get_normalize(self.content[input])
        self.out_transforms = self.get_normalize(self.content[output])

    def get_normalize(self,variables):
        # normalize_mean_50.npz: File storing the mean value of each layer for normalisation.
        normalize_mean = dict(np.load(os.path.join(file_path,"normalize_mean_50.npz")))
        mean = []
        for var in variables:
            if var != "sst":
                mean.append(normalize_mean[var])
            else:
                mean.append(normalize_mean[var]-273.15)
        normalize_mean = np.concatenate(mean)
        # normalize_std_50.npz: File storing the standard deviation of each layer for normalisation.
        normalize_std = dict(np.load(os.path.join(file_path,"normalize_std_50.npz")))
        normalize_std = np.concatenate([normalize_std[var] for var in variables])
        return transforms.Normalize(normalize_mean, normalize_std)

    def read_configs(self):
        file_path = self.yaml_path
        with open(file_path,'r',encoding='utf-8') as f:
            file_content = f.read()
        content = yaml.load(file_content,yaml.FullLoader)
        return content['data']
    
    def create_var_map(self):
        variables = self.content[self.origin_input]
        var_map={}
        idx=0
        for var in variables:
            var_map[var]=idx
            idx+=1
        return var_map
    
    def read_data(self,x):
        var_map = self.create_var_map()
        index_in = []
        for var in self.content[self.input]:
            index_in.append(var_map[var])
        data_x = x[:,index_in,:,:]
        data_x[np.isnan(data_x).bool()] = -32767
        mask = data_x<-30000
        data_x = self.transforms(data_x)
        data_x[mask] = 0
        return data_x.numpy().astype(np.float32)
    
    def get_denormalize(self):
        normalization = self.out_transforms
        mean_norm, std_norm = normalization.mean, normalization.std
        mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
        return transforms.Normalize(mean_denorm,std_denorm)
