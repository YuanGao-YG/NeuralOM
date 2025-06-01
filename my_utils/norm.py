import logging
import glob
from types import new_class
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
import h5py
import math
import torchvision.transforms.functional as TF
# import matplotlib
# import matplotlib.pyplot as plt

class PeriodicPad2d(nn.Module):
    """ 
        pad longitudinal (left-right) circular 
        and pad latitude (top-bottom) with zeros
    """
    def __init__(self, pad_width):
       super(PeriodicPad2d, self).__init__()
       self.pad_width = pad_width

    def forward(self, x):
        # pad left and right circular
        out = F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode="circular") 
        # pad top and bottom zeros
        out = F.pad(out, (0, 0, self.pad_width, self.pad_width), mode="constant", value=0) 
        return out

def reshape_fields(img, inp_or_tar, params, train, normalize=True, orog=None, add_noise=False):
    # Takes in np array of size (n_history+1, c, h, w) 
    # returns torch tensor of size ((n_channels*(n_history+1), crop_size_x, crop_size_y)

    if len(np.shape(img)) == 3:
        img = np.expand_dims(img, 0)
    
    if np.shape(img)[2] == 721:
        img = img[:,:, 0:720, :] # remove last pixel

    n_history = np.shape(img)[0] - 1
    img_shape_x = np.shape(img)[-2]
    img_shape_y = np.shape(img)[-1]
    n_channels = np.shape(img)[1] # this will either be N_in_channels or N_out_channels
    
    if inp_or_tar == 'inp':
        channels = params.in_channels
    elif inp_or_tar == 'ocean':
        channels = params.ocean_channels
    elif inp_or_tar == 'force':
        channels = params.atmos_channels
    else:
        channels = params.out_channels

    if normalize and params.normalization == 'minmax':
        maxs = np.load(params.global_maxs_path)[:, channels]
        mins = np.load(params.global_mins_path)[:, channels]
        img = (img - mins) / (maxs - mins)

    if normalize and params.normalization == 'zscore':
        means = np.load(params.global_means_path)[:, channels]
        stds = np.load(params.global_stds_path)[:, channels]
        img -=means
        img /=stds

    if normalize and params.normalization == 'zscore_lat':
        means = np.load(params.global_lat_means_path)[:, channels,:720]
        stds = np.load(params.global_lat_stds_path)[:, channels,:720]
        img -=means
        img /=stds

    if params.orography and inp_or_tar == 'inp':
        # print('img:', img.shape, 'orog:', orog.shape)
        orog = np.expand_dims(orog, axis = (0,1))
        orog = np.repeat(orog, repeats=img.shape[0], axis=0)
        # print('img:', img.shape, 'orog:', orog.shape)
        img = np.concatenate((img, orog), axis = 1)
        n_channels += 1

    img = np.squeeze(img)
    # if inp_or_tar == 'inp':
    #     img = np.reshape(img, (n_channels*(n_history+1))) # ??
    # elif inp_or_tar == 'tar':
    #     img = np.reshape(img, (n_channels, crop_size_x, crop_size_y)) #??

    if add_noise:
        img = img + np.random.normal(0, scale=params.noise_std, size=img.shape)

    return torch.as_tensor(img)

def vis_precip(fields):
    pred, tar = fields
    fig, ax = plt.subplots(1, 2, figsize=(24,12))
    ax[0].imshow(pred, cmap="coolwarm")
    ax[0].set_title("tp pred")
    ax[1].imshow(tar, cmap="coolwarm")
    ax[1].set_title("tp tar")
    fig.tight_layout()
    return fig

def read_max_min_value(min_max_val_file_path):
    with h5py.File(min_max_val_file_path, 'r') as f:
        max_values = f['max_values']
        min_values = f['min_values']
    return max_values, min_values
    
    



