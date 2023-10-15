from data.data_loader import Dataset_MTS
from exp.exp_basic import Exp_Basic


from utils.metrics import metric

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


import warnings
warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
    
    def _get_data(self, flag):
        args = self.args

        if flag == 'test':
            shuffle_flag = False 
            batch_size = args.batch_size
        elif flag == 'train':
            shuffle_flag = True  
            batch_size = args.forget_num
        else:
            shuffle_flag = True  
            batch_size = args.batch_size
        data_set = Dataset_MTS(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.in_len, args.out_len],  
            data_split = args.data_split,
            std_factor = args.std_factor,
            filter_signal = args.filter_signal,
        )

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers)

        return data_set, data_loader


    def load_train_data(self):
        train_data, train_loader = self._get_data(flag = 'train')
        with torch.no_grad():
            for (mean,std), seq_y, seq_y_std in train_loader:
                self.model.load_params((mean.to(self.device),std.to(self.device)), seq_y.to(self.device))
                break

    def test(self, setting, save_pred = False, inverse = False):
        test_data, test_loader = self._get_data(flag='test')

        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
    
        self.model.eval()
                
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_y_std) in tqdm(enumerate(test_loader)):
                pred =  self._process_one_test_batch(test_data.minmax_inverse_transform, batch_x) 
                true = batch_y_std

                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                if (save_pred):
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())


        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))
        
        if (save_pred):
            # result save
            folder_path = './results/' + setting +'/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))

            preds = np.concatenate(preds, axis = 0)
            trues = np.concatenate(trues, axis = 0)
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)


        return mse, mae


    def _process_one_test_batch(self, minmax_inverse_transform, batch_x):

        logits = (batch_x[0].float().to(self.device), batch_x[1].float().to(self.device))

        outputs = self.model(logits)
        outputs = minmax_inverse_transform(outputs)

        return outputs