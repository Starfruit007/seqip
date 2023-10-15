import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from utils.tools import MinMaxScaler,StandardScaler
from scipy import signal


import warnings
warnings.filterwarnings('ignore')

class Dataset_MTS(Dataset):
    def __init__(self, root_path, data_path='ETTh1.csv', flag='train', size=None, 
                  data_split = [0.7, 0.1, 0.2], std_factor=0.5,filter_signal = False):
        # info
        self.in_len = size[0]
        self.out_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.std_factor = std_factor
        self.filter_signal = filter_signal
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        if (self.data_split[0] > 1):
            train_num = self.data_split[0]; val_num = self.data_split[1]; test_num = self.data_split[2];
        else:
            train_num = int(len(df_raw)*self.data_split[0]); 
            test_num = int(len(df_raw)*self.data_split[2])
            val_num = len(df_raw) - train_num - test_num; 
        border1s = [0, train_num - self.in_len, train_num + val_num - self.in_len]
        border2s = [train_num, train_num+val_num, train_num + val_num + test_num]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

  
        # Get parameters of StandardScaler and MinMaxScaler based on training dataset.
        self.scaler_std = StandardScaler()
        self.scaler_minmax = MinMaxScaler()
        train_data = df_data.values[border1s[0]:border2s[0]]
        self.scaler_std.fit(train_data) 
        train_data_std = self.scaler_std.transform(train_data)
        self.scaler_minmax.fit(train_data_std) 

        # self.std = self.scaler_minmax.transform(train_data_std).std(0)   

        # Perform transform with fitted Scaler.
        data_std = self.scaler_std.transform(df_data.values)
        data = self.scaler_minmax.transform(data_std)

        if self.filter_signal:  self.data_x = signal.savgol_filter(data[border1:border2],9,2,axis=0)  # not activated.
        else: self.data_x = data[border1:border2]
 
        self.data_y = self.data_x
        self.data_y_std = data_std[border1:border2]



    def __generate_guassian_std__(self,mean):
        seq_len, num_sig = mean.shape[-2], mean.shape[-1]
        # log_var = np.log((self.std_factor*self.std)**2)
        # return np.tile(log_var,[seq_len,1])
        return np.ones([seq_len,num_sig])*2*np.log(self.std_factor)
        

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.in_len
        r_begin = s_end
        r_end = r_begin + self.out_len

        mean = self.data_x[s_begin:s_end]
        std = self.__generate_guassian_std__(mean)

        seq_y = self.data_y[r_begin:r_end]
        seq_y_std = self.data_y_std[r_begin:r_end]

        return (mean,std), seq_y, seq_y_std
    
    def __len__(self):
        return len(self.data_x) - self.in_len- self.out_len + 1

    def minmax_inverse_transform(self, data):
        return self.scaler_minmax.inverse_transform(data)

    
    def std_inverse_transform(self, data):
        return self.scaler_std.inverse_transform(data)