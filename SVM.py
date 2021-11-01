from sklearn.svm import SVC
from dataloader import get_dataloader
from vgg import vgg19_bn
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
# y=w*x+b
if __name__=='__main__':
    vgg=vgg19_bn()
    for i in range(40):
        svm=SVC()
        print('------{} svm training-----'.format(i))
        trainloader, testloader, msg = get_dataloader()
        for batchid, batch in enumerate(tqdm(trainloader())):
            x, l = batch #x是人脸图 ，l 是属性
            feature=vgg(x)
            svm.fit(feature, l[:, i])

