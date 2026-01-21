import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn
from icecream import ic
from functools import partial
from torch.autograd import Function
from torch.nn import Module, ModuleList, Sequential



#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def safe_div(num, den, eps = 1e-10):
    return num / den.clamp(min = eps)

# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x.float()

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x.float()

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x


class Momentum_Conservation(object):
    def __init__(self, size_average=True, reduction=True):
        self.reduction = reduction
        self.size_average = size_average

    def calc(self, topo, x, y):
        # SSH = SSH - topo
        # sum(SSH)*(v^2 + u^2)_label - sum(SSH)*(v^2 + u^2)_pred = 0 
        # channels: t,s,u,v,ssh
        # x: num_examples, channels, w, h 
        # ic(x.size(), y.size())
        num_examples = x.size()[0]
        num_channels = x.size()[1]

        topo = topo.view(-1)
        # ic(topo.shape)

        x = x.view(num_examples, num_channels, -1)
        y = y.view(num_examples, num_channels, -1)
        # ic(x.size(), y.size())

        u2_x = torch.pow(x[:,2], 2)
        v2_x = torch.pow(x[:,3], 2) 
        ssh_x = x[:,4] - topo
        x_ = ssh_x * (u2_x + v2_x)
        # ic(u2_x.shape, v2_x.shape, ssh_x.shape, x_.shape)

        u2_y = torch.pow(y[:,2], 2)
        v2_y = torch.pow(y[:,3], 2) 
        ssh_y = y[:,4] - topo
        y_ = ssh_y * (u2_y + v2_y)

        diff_norms = torch.norm(x_.reshape(num_examples,-1) - y_.reshape(num_examples,-1), 2, 1)
        y_norms = torch.norm(y_.reshape(num_examples,-1), 2, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms


    def __call__(self, topo, x, y):
        return self.calc(topo, x, y)

class LpLoss_region_weighted(object):
    def __init__(self, region_idx, region_weight=0, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss_region_weighted, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.region_idx = region_idx
        self.region_weight = region_weight
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]

        weight = torch.zeros_like( x , device=x.device)
        weight = torch.add( weight, 1-self.region_weight )
        weight[:,:,self.region_idx[0]:self.region_idx[1], self.region_idx[2]:self.region_idx[3]] = self.region_weight 

        diff = x - y
        diff = diff * weight
        del weight

        # diff_norms = torch.norm( x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1 )
        diff_norms = torch.norm( diff.reshape(num_examples,-1), self.p, 1 )
        y_norms = torch.norm( y.reshape(num_examples,-1), self.p, 1 )

        tmp = diff_norms/y_norms

        if self.reduction:
            if self.size_average:
                return torch.mean(tmp)
            else:
                return torch.sum(tmp)

        return tmp

    def __call__(self, x, y):
        return self.rel(x, y)


# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        # [B]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class channel_wise_LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True, scale=False):
        super(channel_wise_LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.scale = scale
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        num_channels = x.size()[1]

        x = x.reshape(num_examples, num_channels, -1)
        y = y.reshape(num_examples, num_channels, -1)

        # [B, C]
        diff_norms = torch.norm(x.reshape(num_examples, num_channels, -1) - y.reshape(num_examples, num_channels, -1), self.p, 2)

        y_norms = torch.norm(y.reshape(num_examples, num_channels, -1), self.p, 2)

        if self.reduction:
            if self.size_average:
                if self.scale:
                    channel_wise_mean = torch.mean(diff_norms/y_norms, 0) # [C]: Li, i=1,2,3,...,C
                    channel_mean = torch.mean(diff_norms/y_norms) # scaler 

                    scale_w = channel_mean / channel_wise_mean # [C]: L1/Li, i=1,2,3,...,C
                    channel_scale = torch.mean(scale_w * channel_wise_mean) # \sum w_i*L_i
                    return channel_scale, channel_wise_mean * scale_w 
                else:
                    channel_mean = torch.mean(diff_norms/y_norms, 0)
                    return torch.mean(diff_norms/y_norms), channel_mean
            else:
                if self.scale:
                    channel_sum = torch.sum(diff_norms/y_norms, 0)
                    scale_w = channel_sum[0] / channel_sum
                    channel_sum_scale = torch.sum(scale_w * channel_sum)
                    return channel_sum_scale, channel_sum*scale_w
                else:
                    channel_sum = torch.sum(diff_norms/y_norms, 0)
                    return torch.sum(diff_norms/y_norms), channel_sum
        else:
            return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss

# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

class LossScaleFunction(Function):
    """
    refer to MetNet-3
    """
    @staticmethod
    def forward(ctx, x, eps):
        ctx.eps = eps
        assert x.ndim == 4
        return x

    @staticmethod
    def backward(ctx, grads):
        num_channels = grads.shape[1]

        safe_div_ = partial(safe_div, eps = ctx.eps)

        weight = safe_div_(1., grads.norm(p = 2, keepdim = True, dim = (-1, -2)))
        l1_normed_weight = safe_div_(weight, weight.sum(keepdim = True, dim = 1))

        scaled_grads = num_channels * l1_normed_weight * grads

        return scaled_grads, None

class LossScaler(Module):
    """
    refer to MetNet-3
    """
    def __init__(self, eps = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return LossScaleFunction.apply(x, self.eps)
