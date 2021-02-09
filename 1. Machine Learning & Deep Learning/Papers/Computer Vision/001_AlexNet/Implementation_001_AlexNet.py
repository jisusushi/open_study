import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# define pytorch device
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define model parameters
NUM_EPOCHS= 90
BATCH_SIZE= 128
MOMENTUM= 0.9
LR_DECAY= 0.0005
LR_INIT= 0.01
IMAGE_DIM= 227
NUM_CLASSES= 1000
DEVICE_IDS= [0, 1, 2, 3]

# ! I didn't downloaded the dataset(it's over 100G), so the directory is fake. I didn't run this code actually.
INPUT_ROOT_DIR= 'alexnet_data_in'
TRAIN_IMG_DIR= 'alexnet_data_in/imagenet'
OUTPUT_DIR= 'alexnet_data_out'
LOG_DIR= OUTPUT_DIR + '/tblogs'
CHECKPOINT_DIR= OUTPUT_DIR + '/models'

# make checkopoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class AlexNet(nn.Module):
    def  __init__(self, num_classes= 1000):
        super(AlexNet, self).__init__()

        # the input size would be (batch x 3 x 227 x 227)
        # ! width and height was 224 in original paper, but we have to set it 227 to set 2nd conv. layer as 55 x 55
        self.net= nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier= nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256*6*6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

        self.init_bias()

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std= 0.01)
                nn.init.constant_(layer.bias, 0)

        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        x= self.net(x)
        x= x.view(-1, 256*6*6)
        return self.classifier(x)


if __name__ == '__main__':
    alexnet= AlexNet(num_classes=NUM_CLASSES).to(device)
    # train alexnet in multiple GPUs
    alexnet= torch.nn.parallel.DataParallel(alexnet, device_ids= DEVICE_IDS)

    # create dataset and dataloader
    datasets= datasets.ImageFolder(TRAIN_IMG_DIR, transforms.Compose([
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

    dataloader= data.DataLoader(
        datasets,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        batch_size=BATCH_SIZE
    )

    # create optimizer
    optimizer= optim.Adam(params=alexnet.parameters(), lr=0.0001)

    lr_scheduler= optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


    # start training
    total_steps= 1
    for epoch in range(NUM_EPOCHS):
        lr_scheduler.step()
        for imgs, classes in dataloader:
            imgs.to(device)
            classes.to(device)

            output= alexnet(imgs)
            loss= F.cross_entropy(output, classes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if total_steps % 10 == 0:
                with torch.no_grad():
                    _, preds= torch.max(output, 1)
                    accuracy= torch.sum(preds == classes)
                    print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                          .format(epoch + 1, total_steps, loss.item(), accuracy.item()))

            if total_steps % 100 == 0:
                with torch.no_grad():
                    print('*' * 10)
                    for name, parameter in alexnet.named_parameters():
                        if parameter.grad is not None:
                            avg_grad= torch.mean(parameter.grad)
                            print('\t{} - grad_avg: {}'.format(name, avg_grad))
                        if parameter.data is not None:
                            avg_weight= torch.mean(parameter.data)
                            print('\t{} - param_avg: {}'.format(name, avg_weight))

            total_steps+=1
            