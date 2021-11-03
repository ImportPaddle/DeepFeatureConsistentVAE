import paddle
import datetime
import logging
import os
import torch
import paddle.nn.functional as F
import warnings
import glob
import yaml
import  numpy as np
if __name__=='__main__':
    np.random.seed(1)
    mu=np.random.rand(144,128)
    log_var=np.random.rand(144,128)

    mu=torch.tensor(mu)
    log_var=torch.tensor(log_var)

    mu=paddle.to_tensor(np.array(mu))
    log_var=paddle.to_tensor(np.array(log_var))

    way1=F.kl_div(mu,log_var, reduction='mean')
    way2=-0.5 * paddle.mean(1 + log_var - mu ** 2 - log_var.exp())
    way3=paddle.mean(-0.5 * paddle.sum(1 + log_var - mu ** 2 - log_var.exp(), axis=1), axis=0)

    print(way1)
    print(way2)
    print(way3)

