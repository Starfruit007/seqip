import numpy as np
import torch
import json

class StandardScaler():
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean

class MinMaxScaler():
    def __init__(self, min_val = 0, diff_val = 1):
        self.min_val = min_val
        self.diff_val = diff_val

    def fit(self,data):
        self.min_val = data.min(0)
        self.max_val = data.max(0)
        self.diff_val = self.max_val - self.min_val

    def transform(self,data):
        min_val = torch.from_numpy(self.min_val).type_as(data).to(data.device) if torch.is_tensor(data) else self.min_val
        diff_val = torch.from_numpy(self.diff_val).type_as(data).to(data.device) if torch.is_tensor(data) else self.diff_val
        return (data-min_val) / diff_val

    def inverse_transform(self,data):
        min_val = torch.from_numpy(self.min_val).type_as(data).to(data.device) if torch.is_tensor(data) else self.min_val
        diff_val = torch.from_numpy(self.diff_val).type_as(data).to(data.device) if torch.is_tensor(data) else self.diff_val 
        return (data*diff_val) + min_val

def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)
    return args

def string_split(str_for_split):
    str_no_space = str_for_split.replace(' ', '')
    str_split = str_no_space.split(',')
    value_list = [eval(x) for x in str_split]

    return value_list
