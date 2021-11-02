import paddle
import datetime
import logging
import os
import paddle.nn.functional as F
import warnings
import glob
import yaml
import  numpy as np
if __name__=='__main__':
    mu=np.random.rand(144,128)
    log_var=np.random.rand(144,128)

    mu=torch.tensor(mu)
    log_var=torch.tensor(log_var)

    t=torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    mu=paddle.to_tensor(mu)
    log_var=paddle.to_tensor(log_var)
    p=paddle.mean(-0.5 * paddle.sum(1 + log_var - mu ** 2 - log_var.exp(), axis=1), axis=0)
    print(t)
    print(p)

