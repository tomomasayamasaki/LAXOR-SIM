import numpy as np
import torch
import torch.nn.functional as F

def load_npys(w_path, bias_path, BNbeta_path=None, BNgamma_path=None, BNmean_path=None, BNstd_path=None):
    """
        load npy files
    """
    weights = np.load(w_path)
    bias = np.load(bias_path)
    if not BNbeta_path == None:
        BN_beta = np.load(BNbeta_path)
        BN_gamma = np.load(BNgamma_path)
        BN_mean = np.load(BNmean_path)
        BN_std = np.load(BNstd_path)
        return weights, bias, BN_beta, BN_gamma, BN_mean, BN_std

    return weights, bias

def binaryhard2soft(array):
    """
        convert 1-0 binary to 1-(-1) binary
    """
    arr = np.where(array==1, 1, -1)
    return arr

def binarize_softbase(array):
    """
        binarize array with 1 or -1, which is software base
    """
    arr = np.where(array>=0, 1, -1)
    return arr

def binarize_hardbase(array):
    """
        binarize array with 1 or 0, which is hardware base
    """
    arr = np.where(array>=0, 1, 0)
    return arr

def numpy2torch(array):
    """
        numpy.darray is changed to torch.tensor
    """
    arr = torch.from_numpy(array)
    return arr.float()

def Fconv2D(input, weights, bias, stride=1, padding=0):
    """
        convolution by pytorch
    """
    out = F.conv2d(numpy2torch(input), numpy2torch(weights), numpy2torch(bias), stride=stride, padding=padding)
    return out

def Fmaxpooling(input, ksize, stride):
    out = F.max_pool2d(numpy2torch(input), ksize, stride)
    out = np.array(out)
    return out

def Favepooling(input, ksize, stride):
    out = F.avg_pool2d(numpy2torch(input), ksize, stride)
    out = np.array(out)
    return out

def filpWeight_lasagne(array):
    """
        re-flip weights trained by lasagne & theano package

        |---|---|           |---|---|
        | 0 | 1 |           | 3 | 2 |
        |---|---|    ->     |---|---|
        | 2 | 3 |　　　　　　 | 1 | 0 |
        |---|---|           |---|---|
    """
    return array[:, :, ::-1, ::-1].copy()

def inverseStd_lasagne(array):
    """
        If you trained your model by theano,
        the std parameter of batach normalization from theano is inversed and includes epsilon.

        function
        ---------
        sqrt(std^2+epsilon) = 1/std_theano


        function on thean
        ---------
        std_theano = 1/np.sqrt(std^2+epsilon)
    """
    l = array.shape
    ones = np.ones(l)
    return ones/array
