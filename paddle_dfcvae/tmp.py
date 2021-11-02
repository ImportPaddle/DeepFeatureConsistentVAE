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

    t=torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    mu=paddle.to_tensor(np.array(mu))
    log_var=paddle.to_tensor(np.array(log_var))
    p=0.5 * paddle.sum(paddle.exp(log_var)+mu ** 2 -1 - log_var,axis=1)
    print(p)
    p=paddle.mean(p)
    print(p)
    p=paddle.mean(-0.5 * paddle.sum(1 + log_var - mu ** 2 - log_var.exp(), axis=1), axis=0)
    print(p)
    kl=F.kl_div(mu, log_var, reduction='mean')

    print(t)
    print(p)
    print(kl)

