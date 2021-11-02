import math

from tqdm import tqdm
import paddle
import datetime
import logging
import os
from dataloader import get_dataloader
from architectures import get_model
from optimizer import get_optimizers
import paddle.nn.functional as F
import warnings
import glob
import yaml
from PIL import Image
from dataloader import get_dataloader
import numpy as np
def setdir(path):
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
def test():
    path='./experiments/dfcvae-latent-100-conv-featureLoss_KLdLoss/ckpt/600.pdparams'
    state=paddle.load(path)['models']['dfcvae']
    model,msg=get_model()
    model.set_state_dict(state)
    trainDataloader,valDataloader,img_cnt,msg=get_dataloader()

    setdir('./imgGenerateResult')
    for batch_id, batch in enumerate(tqdm(valDataloader())):
        x,_=batch
        output=model(x)
        sourceImg=np.array((x[0]+1)/2*255,dtype='uint8')
        # print(sourceImg)
        image=Image.fromarray(sourceImg.transpose([1,2,0]))
        image.save('./imgGenerateResult/{}.png'.format(batch_id))


        # reconImg=np.array(output+1*255,dtype='uint8') #
        reconImg = np.array((output[0][0]+1)/2*255,dtype='uint8').transpose([1,2,0])
        # print(reconImg)
        # print(reconImg.shape)
        # print(reconImg[:,:,0]) #128
        # print(reconImg[:, :, 1]) #121
        print(reconImg[:, :, 2])
        image = Image.fromarray(reconImg)
        image.save('./imgGenerateResult/{}-rec.png'.format(batch_id))

        if batch_id>100:
            break

if __name__=='__main__':
    test()