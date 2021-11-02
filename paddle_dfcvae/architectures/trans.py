import  paddle
import torch
from vgg import vgg19_bn
def isIn(key, str):
    for ele in key:
        if ele in str:
            return True
def rename_state_dict(paddle_dict, torch_dict):
    result = {}
    _ = {}
    skip_params = []
    for k, v in torch_dict.items():
        if isIn(skip_params, k):
            print('---skip batches_tracked---')
        else:

            _[k] = v.cpu().detach().numpy()
    torch_dict = _

    assert len(torch_dict) == len(paddle_dict)
    for paddle_param, torch_param in zip(paddle_dict.items(), torch_dict.items()):
        k1, v1 = paddle_param
        k2, v2 = torch_param
        v1 = v1.numpy()
        v2 = v2
        if 'classifier' in k1:
            v2=v2.T
        print('{} shape {}, {} shape {}'.format(k1, v1.shape, k2, v2.shape))
        assert v1.shape == v2.shape
        result[k1] = v2
    print(len(torch_dict))
    print(len(paddle_dict))
    return result
def write_dict(state_dict,name):
    lines=[]
    for k,v in state_dict.items():
        if 'batches_tracked' in k:
            print('---skip--batches_tracked-')
            continue
        try:
            line=str(k)+'\t'+str(v.cpu().detach().numpy().shape)+'\n'
        except:
            line = str(k) + '\t' + str(v.shape) + '\n'
        # line=str(v)+'\t'+str(v.cpu().detach().numpy().shape)+'\n'
        lines.append(line)
    with open(name,'w')as f:
        f.writelines(lines)

def trans():
    path = './pretrained/vgg19_bn-c79401a0.pth'
    torch_dict = torch.load(path)
    paddle_dict = {}
    paddle_dict['state_dict'] = {}

    model = vgg19_bn(pretrained=False)
    write_dict(model.state_dict(), './paddleParams.txt')
    write_dict(torch_dict, './torchParams.txt')

    paddle_dict['state_dict'] = rename_state_dict(model.state_dict(), torch_dict)
    write_dict(paddle_dict['state_dict'], './rename_Params.txt')
    model.set_state_dict(paddle_dict['state_dict'])
    paddle.save(paddle_dict,'./pretrained/vgg19_bn-c79401a0.pdparams')
def see():
    print(1)
    path = './pretrained/vgg19_bn-c79401a0.pth'
    torch_dict = torch.load(path)
    print(torch_dict.keys())
if __name__=='__main__':
    trans()