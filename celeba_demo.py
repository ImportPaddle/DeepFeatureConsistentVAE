from paddle.io import DataLoader
import paddle.vision.transforms as Transforms


from CelebaDataset import AttribDataset

path = "/Users/xuguan/Desktop/repo-list/DeepFeatureConsistentVAE/dataset/celeba/celeba_demo"
def train_dataloader(transform=None):
    dataset = AttribDataset(pathdb=path, transform=transform)

    num_train_imgs = len(dataset)
    print(num_train_imgs)
    return DataLoader(dataset,
                      batch_size=8,
                      shuffle=True,
                      drop_last=True)


if __name__ == '__main__':
    transform = Transforms.ToTensor()
    data_loader = train_dataloader(transform)
    for (img, attr) in data_loader:
        print(f'img shape: {img.shape}, attr: {attr.shape}')
