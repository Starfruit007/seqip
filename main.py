import argparse
import torch

from exp.exp_main import Exp_Main
from utils.tools import string_split
import numpy as np

parser = argparse.ArgumentParser(description='Sequential Indeterminate Probability')

parser.add_argument('--data', type=str, required=False, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')  
parser.add_argument('--data_split', type=str, default='0.7,0.1,0.2',help='train/val/test split, can be ratio or number')


parser.add_argument('--in_len', type=int, default=48, help='input MTS length (T)')
parser.add_argument('--out_len', type=int, default=168, help='output MTS length (\tau)')
parser.add_argument('--data_dim', type=int, default=7, help='Number of dimensions of the MTS data (D)')
parser.add_argument('--std_factor', type=float, default=0.4, help='Std factor ')
parser.add_argument('--filter_signal', action='store_true', help='whether to filter the signals', default=False)

parser.add_argument('--forget_num', type=int, default=5000, help='forget number')
parser.add_argument('--stable_num', type=float, default=1e-20, help='stable number')
parser.add_argument('--monte_carlo_num', type=int, default=8, help='monte carlo number')
parser.add_argument('--top_p', type=float, default=None, help='approximate latent space.')


parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
parser.add_argument('--itr', type=int, default=1, help='experiments times')

parser.add_argument('--save_pred', action='store_true', help='whether to save the predicted future MTS', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
    print(args.gpu)

data_parser = {
    'ETTh1':{'data':'ETT-small/ETTh1.csv', 'data_dim':7, 'split':[12*30*24, 4*30*24, 4*30*24]}, #[12*30*24, 4*30*24, 4*30*24]
    'ETTm1':{'data':'ETT-small/ETTm1.csv', 'data_dim':7, 'split':[4*12*30*24, 4*4*30*24, 4*4*30*24]},
    'WTH':{'data':'weather/weather.csv', 'data_dim':21, 'split':[28*30*24, 10*30*24, 10*30*24]},
    'ECL':{'data':'electricity/electricity.csv', 'data_dim':321, 'split':[15*30*24, 3*30*24, 4*30*24]},
    'ILI':{'data':'illness/national_illness.csv', 'data_dim':7, 'split':[0.7, 0.1, 0.2]},
    'Traffic':{'data':'traffic/traffic.csv', 'data_dim':862, 'split':[0.7, 0.1, 0.2]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.data_dim = data_info['data_dim']
    args.data_split = data_info['split']
else:
    args.data_split = string_split(args.data_split)

print('Args in experiment:')
print(args)

Exp = Exp_Main

mses,maes = [],[]
for ii in range(args.itr):
    # setting record of experiments
    setting = 'Seqip_{}_il{}_ol{}_std{}_itr{}-{}'.format(args.data, 
                args.in_len, args.out_len, args.std_factor,args.itr,ii+1)

    exp = Exp(args) # set experiments
    exp.load_train_data()
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mse, mae = exp.test(setting, args.save_pred)
    mses.append(mse)
    maes.append(mae)

print_txt = 'mses: {} mean: {}; maes: {} mean: {}.'.format(mses, np.mean(mses), maes,np.mean(maes))
print(print_txt)
