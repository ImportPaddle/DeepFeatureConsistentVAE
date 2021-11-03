import numpy as np
import os
import sys
DIR=os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(DIR,'../../'))

from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger


#import torch RotNet
from torch_dfcvae.models import  vae_models
from paddle_dfcvae.architectures import  get_model
import torch,os
import paddle


SEED=100
torch.manual_seed(SEED)
paddle.seed(SEED)
np.random.seed(SEED)

def paddleRes(net, data,out_feat_keys=None):
    net.eval()
    data = paddle.to_tensor(data)
    if out_feat_keys:
        res = net(data,out_feat_keys=out_feat_keys)
    else:
        res = net(data)
    return res.numpy()


def torchRes(net, data,out_feat_keys=None):
    net.eval()
    data = torch.from_numpy(data)
    if out_feat_keys:
        res = net(data,out_feat_keys=out_feat_keys)
    else:
        res = net(data)
    return res.data.cpu().numpy()


# def get():
#     opt = {'num_classes': 4, 'num_stages': 4, 'use_avg_on_conv3': False}
#     model_ext_pytorch = NetworkInNetwork.create_model(opt)
#     torch.save(model_ext_pytorch.state_dict(),"./model_ext_pytorch.pth")
#
#     opt = {'num_classes': 10, 'nChannels': 192, 'cls_type': 'NIN_ConvBlock3'}
#     model_cla = NonLinearClassifier.create_model(opt)  # out 128,10
#     torch.save(model_cla.state_dict(), './model_cla_pytorch.pth')
# def trans():
#     path = './model_ext_pytorch.pth'
#     torch_dict = torch.load(path)
#     paddle_dict = {}
#     for key in torch_dict:
#         weight = torch_dict[key].cpu().detach().numpy()
#         # print(key)
#         if key == 'fc.weight' or key == '_feature_blocks.4.Classifier.weight':
#             weight = weight.transpose()
#         key = key.replace('running_mean', '_mean')
#         key = key.replace('running_var', '_variance')
#         paddle_dict[key] = weight
#     paddle.save(paddle_dict, './model_ext_paddle.pdparams')
#
#     path = './model_cla_pytorch.pth'
#     torch_dict = torch.load(path)
#     paddle_dict = {}
#     for key in torch_dict:
#         weight = torch_dict[key].cpu().detach().numpy()
#         # print(key)
#         if key == 'fc.weight' or key == 'classifier.Liniear_F.weight':
#             weight = weight.transpose()
#         key = key.replace('running_mean', '_mean')
#         key = key.replace('running_var', '_variance')
#         paddle_dict[key] = weight
#     paddle.save(paddle_dict, './model_cla_paddle.pdparams')

def tranState(torchState):
    pass
def main():
    reprod_log_1 = ReprodLogger()
    reprod_log_2 = ReprodLogger()
    fake_data = np.random.rand(128, 3, 32, 32).astype(np.float32)

    ##torch cfcvae
    option= {'name': 'DFCVAE','in_channels': 3,'latent_dim': 100}
    dfcvae_pytorch = vae_models['DFCVAE'](**option)

    ##paddle cfcvae
    dfcvae_paddle = get_model()
    paddle_dict=tranState(dfcvae_pytorch.state_dict())
    dfcvae_paddle.set_state_dict(paddle_dict)


    pytorch_res = torchRes(dfcvae_pytorch,fake_data)
    paddle_res = paddleRes(dfcvae_paddle,fake_data)


    reprod_log_1.add("model_torch", pytorch_res)
    reprod_log_1.save("net_torch.npy")

    reprod_log_2.add("model_paddle", paddle_res)
    reprod_log_2.save("net_paddle.npy")



def check():
    diff_helper = ReprodDiffHelper()
    info1 = diff_helper.load_info("./net_torch.npy")
    info2 = diff_helper.load_info("./net_paddle.npy")

    diff_helper.compare_info(info1, info2)
    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path="./diff-model.txt")


if __name__ == "__main__":
    main()
    check()