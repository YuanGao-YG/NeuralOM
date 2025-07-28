import os
import sys
import time
import glob
import h5py
import logging
import argparse
import numpy as np
from icecream import ic
from datetime import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from my_utils.YParams import YParams
from my_utils.data_loader import get_data_loader
from my_utils import logging_utils
logging_utils.config_logger()


def load_model(model, params, checkpoint_file):
    model.zero_grad()
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname)
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val  
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

def setup(params):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # get data loader
    valid_data_loader, valid_dataset = get_data_loader(params, params.test_data_path, dist.is_initialized(), train=False)

    img_shape_x = valid_dataset.img_shape_x
    img_shape_y = valid_dataset.img_shape_y
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y

    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)

    params['N_in_channels'] = n_in_channels
    params['N_out_channels'] = n_out_channels

    if params.normalization == 'zscore': 
        params.means = np.load(params.global_means_path)
        params.stds = np.load(params.global_stds_path)

    if params.nettype == 'NeuralOM':
        from networks.MIGNN1 import MIGraph as model
        from networks.MIGNN2 import MIGraph_stage2 as model2
    else:
        raise Exception("not implemented")

    checkpoint_file  = params['best_checkpoint_path']
    checkpoint_file2  = params['best_checkpoint_path2']
    logging.info('Loading trained model checkpoint from {}'.format(checkpoint_file))
    logging.info('Loading trained model2 checkpoint from {}'.format(checkpoint_file2))
    
    model = model(params).to(device) 
    model = load_model(model, params, checkpoint_file)
    model = model.to(device)

    print('model is ok')

    model2 = model2(params).to(device) 
    model2 = load_model(model2, params, checkpoint_file2)
    model2 = model2.to(device)

    print('model2 is ok')
    
    files_paths = glob.glob(params.test_data_path + "/*.h5")
    files_paths.sort()

    # which year
    yr = 0
    logging.info('Loading inference data')
    logging.info('Inference data from {}'.format(files_paths[yr]))
    climate_mean = np.load('./data/climate_mean_s_t_ssh.npy')
    valid_data_full = h5py.File(files_paths[yr], 'r')['fields'][:365, :, :, :]
    valid_data_full = valid_data_full - climate_mean

    return valid_data_full, model, model2

    
def autoregressive_inference(params, init_condition, valid_data_full, model, model2): 
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        
    icd = int(init_condition) 
    
    exp_dir = params['experiment_dir'] 
    dt                = int(params.dt)
    prediction_length = int(params.prediction_length/dt)
    n_history      = params.n_history
    img_shape_x    = params.img_shape_x
    img_shape_y    = params.img_shape_y
    in_channels    = np.array(params.in_channels)
    out_channels   = np.array(params.out_channels)
    atmos_channels = np.array(params.atmos_channels)
    n_in_channels  = len(in_channels)
    n_out_channels = len(out_channels)
    
    seq_real        = torch.zeros((prediction_length, n_out_channels, img_shape_x, img_shape_y))
    seq_pred        = torch.zeros((prediction_length, n_out_channels, img_shape_x, img_shape_y))


    valid_data = valid_data_full[icd:(icd+prediction_length*dt+n_history*dt):dt][:, params.in_channels][:,:,0:360]
    logging.info(f'valid_data_full: {valid_data_full.shape}')
    logging.info(f'valid_data: {valid_data.shape}')
    
    # normalize
    if params.normalization == 'zscore': 
        valid_data = (valid_data - params.means[:,params.in_channels])/params.stds[:,params.in_channels]
        valid_data = np.nan_to_num(valid_data, nan=0)
        
    valid_data = torch.as_tensor(valid_data)

    # autoregressive inference
    logging.info('Begin autoregressive inference')
    
    
    with torch.no_grad():
        for i in range(valid_data.shape[0]): 
            if i==0: # start of sequence, t0 --> t0'
                first = valid_data[0:n_history+1]
                ic(valid_data.shape, first.shape)
                future = valid_data[n_history+1]
                ic(future.shape)

                for h in range(n_history+1):
                    
                    seq_real[h] = first[h*n_in_channels : (h+1)*n_in_channels, :93]
                    
                    seq_pred[h] = seq_real[h]

                first = first.to(device, dtype=torch.float)
                first_ocean = first[:, params.ocean_channels, :, :]
                ic(first_ocean.shape)
                future_force0 = first[:, params.atmos_channels, :, :]
               
                future_force = future[params.atmos_channels, :360, :720]
                future_force = torch.unsqueeze(future_force, dim=0).to(device, dtype=torch.float)
                model_input = torch.cat((first_ocean, future_force0, future_force.cuda()), axis=1)
                ic(model_input.shape)
                model1_future_pred = model(model_input)
                with h5py.File(params.land_mask_path, 'r') as _f: 
                    mask_data = torch.as_tensor(_f['fields'][:,out_channels, :360, :720], dtype=bool).to(device, dtype=torch.bool)
                model1_future_pred = torch.masked_fill(input=model1_future_pred, mask=~mask_data, value=0)
                future_pred = model2(model1_future_pred) + model1_future_pred

            else:
                if i < prediction_length-1:
                    future0 = valid_data[n_history+i]
                    future = valid_data[n_history+i+1]

                inf_one_step_start = time.time()
                future_force0 = future0[params.atmos_channels, :360, :720]
                future_force = future[params.atmos_channels, :360, :720]
                future_force0 = torch.unsqueeze(future_force0, dim=0).to(device, dtype=torch.float)
                future_force = torch.unsqueeze(future_force, dim=0).to(device, dtype=torch.float)
                model1_future_pred = model(torch.cat((future_pred.cuda(), future_force0, future_force), axis=1)) #autoregressive step
                with h5py.File(params.land_mask_path, 'r') as _f: 
                    mask_data = torch.as_tensor(_f['fields'][:,out_channels, :360, :720], dtype=bool).to(device, dtype=torch.bool)
                model1_future_pred = torch.masked_fill(input=model1_future_pred, mask=~mask_data, value=0)
                future_pred = model2(model1_future_pred) + model1_future_pred
                inf_one_step_time = time.time() - inf_one_step_start

                logging.info(f'inference one step time: {inf_one_step_time}')

            if i < prediction_length - 1: # not on the last step
                with h5py.File(params.land_mask_path, 'r') as _f: 
                    mask_data = torch.as_tensor(_f['fields'][:,out_channels, :360, :720], dtype=bool)
                seq_pred[n_history+i+1] = torch.masked_fill(input=future_pred.cpu(), mask=~mask_data, value=0)
                seq_real[n_history+i+1] = future[:93]
                history_stack = seq_pred[i+1:i+2+n_history]

            future_pred = history_stack

            pred = torch.unsqueeze(seq_pred[i], 0)
            tar  = torch.unsqueeze(seq_real[i], 0)

            with h5py.File(params.land_mask_path, 'r') as _f: 
                mask_data = torch.as_tensor(_f['fields'][:,out_channels, :360, :720], dtype=bool)
                ic(mask_data.shape, pred.shape, tar.shape)
            pred = torch.masked_fill(input=pred, mask=~mask_data, value=0)
            tar  = torch.masked_fill(input=tar,  mask=~mask_data, value=0)

            print(torch.mean((pred-tar)**2))

    
    seq_real = seq_real * params.stds[:,params.out_channels] + params.means[:,params.out_channels]
    seq_real = seq_real.numpy()
    seq_pred = seq_pred * params.stds[:,params.out_channels] + params.means[:,params.out_channels]
    seq_pred = seq_pred.numpy()
   

    return (np.expand_dims(seq_real[n_history:], 0), 
            np.expand_dims(seq_pred[n_history:], 0), 
           )     


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default='../exp_15_levels', type=str)
    parser.add_argument("--config", default='full_field', type=str)
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--prediction_length", default=61, type=int)
    parser.add_argument("--finetune_dir", default='', type=str)
    parser.add_argument("--ics_type", default='default', type=str)
    args = parser.parse_args()

    config_path = os.path.join(args.exp_dir, args.config, args.run_num, 'config.yaml')
    params = YParams(config_path, args.config)

    params['resuming']           = False
    params['interp']             = 0 
    params['world_size']         = 1
    params['local_rank']         = 0
    params['global_batch_size']  = params.batch_size
    params['prediction_length']  = args.prediction_length
    params['multi_steps_finetune'] = 1

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True

    # Set up directory
    if args.finetune_dir == '':
        expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    else:
        expDir = os.path.join(params.exp_dir, args.config, str(args.run_num), args.finetune_dir)
    logging.info(f'expDir: {expDir}')
    params['experiment_dir']       = expDir 
    params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
    params['best_checkpoint_path2'] = os.path.join(expDir, 'model2/10_steps_finetune/training_checkpoints/best_ckpt.tar') 

    # set up logging
    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'inference.log'))
    logging_utils.log_versions()
    params.log()

    if params["ics_type"] == 'default':
        ics = np.arange(0, 240, 1)
        n_ics = len(ics)
        print('init_condition:', ics)

    logging.info("Inference for {} initial conditions".format(n_ics))

    try:
      autoregressive_inference_filetag = params["inference_file_tag"]
    except:
      autoregressive_inference_filetag = ""
    if params.interp > 0:
        autoregressive_inference_filetag = "_coarse"

    valid_data_full, model, model2 = setup(params)


    seq_pred = []
    seq_real = []

    # run autoregressive inference for multiple initial conditions
    for i, ic_ in enumerate(ics):
        logging.info("Initial condition {} of {}".format(i+1, n_ics))
        seq_real, seq_pred = autoregressive_inference(params, ic_, valid_data_full, model, model2)

        prediction_length = seq_real[0].shape[0]
        n_out_channels = seq_real[0].shape[1]
        img_shape_x = seq_real[0].shape[2]
        img_shape_y = seq_real[0].shape[3]

        # save predictions and loss
        save_path = os.path.join(params['experiment_dir'], 'results_simulation.h5')
        logging.info("Saving to {}".format(save_path))
        print(f'saving to {save_path}')
        if i==0:
            f = h5py.File(save_path, 'w')
            f.create_dataset(
                    "ground_truth",
                    data=seq_real,
                    maxshape=[None, prediction_length, n_out_channels, img_shape_x, img_shape_y], 
                    dtype=np.float32)
            f.create_dataset(
                    "predicted",       
                    data=seq_pred, 
                    maxshape=[None, prediction_length, n_out_channels, img_shape_x, img_shape_y], 
                    dtype=np.float32)
            f.close()
        else:
            f = h5py.File(save_path, 'a')

            f["ground_truth"].resize((f["ground_truth"].shape[0] + 1), axis = 0)
            f["ground_truth"][-1:] = seq_real 

            f["predicted"].resize((f["predicted"].shape[0] + 1), axis = 0)
            f["predicted"][-1:] = seq_pred 
            f.close()

