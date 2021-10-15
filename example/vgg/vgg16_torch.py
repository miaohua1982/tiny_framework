import torch as t
from torchvision import transforms
import torchvision
from torch import optim
from torch import nn

"""
1. load CIFAR10 datasets, 3*32*32 for one picture
"""
ds_path = '/home/miaohua/Documents/Datasets'
batch_size = 16
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root=ds_path, train=True, download=False, transform=transform)
trainloader = t.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.CIFAR10(root=ds_path, train=False, download=False, transform=transform)
testloader = t.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

"""
第二步：定义VGG16 模型，全部的卷积层，也就是特征抽取层和models.vgg16(pretrained=True)是一样的
       但是FC层根据自己的特点自己定义了。
"""
class VGGTest(nn.Module):
    def __init__(self, pretrained=True, numClasses=10):
        super(VGGTest, self).__init__()

        # 100% 还原特征提取层，也就是5层共13个卷积层
        # conv1 1/2
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv2 1/4
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv3 1/8
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv4 1/16
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv5 1/32
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # load pretrained params from torchvision.models.vgg16(pretrained=True)
        # 从原始的 models.vgg16(pretrained=True) 中预设值参数值。
        if pretrained:
            pretrained_model = torchvision.models.vgg16(pretrained=pretrained)  # 从预训练模型加载VGG16网络参数
            pretrained_params = pretrained_model.state_dict()
            keys = list(pretrained_params.keys())
            new_dict = {}
            for index, key in enumerate(self.state_dict().keys()):
                new_dict[key] = pretrained_params[keys[index]]
            self.load_state_dict(new_dict)

        # 但是至于后面的全连接层，根据实际场景，就得自行定义自己的FC层了。
        self.classifier = nn.Sequential(  # 定义自己的分类层
            # 原始模型vgg16输入image大小是224 x 224
            # 我们测试的自己模仿写的模型输入image大小是32 x 32
            # 大小是小了 7 x 7倍
            nn.Linear(in_features=512 * 1 * 1, out_features=256),  # 自定义网络输入后的大小。
            # nn.Linear(in_features=512 * 7 * 7, out_features=256),  # 原始vgg16的大小是 512 * 7 * 7 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=numClasses),
        )

    def forward(self, x):   # output: 32 * 32 * 3
        x = self.relu1_1(self.conv1_1(x))  # output: 32 * 32 * 64
        x = self.relu1_2(self.conv1_2(x))  # output: 32 * 32 * 64
        x = self.pool1(x)  # output: 16 * 16 * 64

        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        x = self.pool2(x)

        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        x = self.pool3(x)

        x = self.relu4_1(self.conv4_1(x))
        x = self.relu4_2(self.conv4_2(x))
        x = self.relu4_3(self.conv4_3(x))
        x = self.pool4(x)

        x = self.relu5_1(self.conv5_1(x))
        x = self.relu5_2(self.conv5_2(x))
        x = self.relu5_3(self.conv5_3(x))
        x = self.pool5(x)

        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output

def accu(pred, target):
    p = t.argmax(pred, dim=1)
    return (p == target).float().mean()

def vgg_test(net, criterion):
    running_loss = 0.0
    running_acc = 0.0
    for data in testloader:
        inputs, labels = data  # labels: [batch_size, 1]

        outputs = net(inputs)  # outputs: [batch_size, 10]
        # loss
        loss = criterion(outputs, labels)
        # acc
        acc = accu(outputs, labels)
        # 打印loss
        running_loss += loss.item()
        running_acc += acc.item()

    print('In test set loss: %.5f, accu: %.5f' % (running_loss/len(testloader), running_acc/len(testloader)))

def vgg_train():
    epochs = 2  # 训练次数
    learning_rate = 1e-4  # 学习率

    net = VGGTest()
    criterion = nn.CrossEntropyLoss() # 交叉熵损失
    optimizer = optim.Adam(net.parameters(), lr=learning_rate) # Adam优化器

    for epoch in range(epochs):  # 迭代
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data  # labels: [batch_size, 1]

            # 初始化梯度
            optimizer.zero_grad()

            outputs = net(inputs)  # outputs: [batch_size, 10]
            # print(outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            acc = accu(outputs, labels)
            # 打印loss
            running_loss += loss.item()
            running_acc += acc.item()

            if i % 20 == 19:  # print loss every 20 mini batch
                print('[%d, %5d] loss: %.5f, accu: %.5f' %
                      (epoch + 1, i + 1, running_loss / 20.0, running_acc / 20.0))
                running_loss = 0.0
                running_acc = 0.0
        vgg_test(net, criterion)

    print('Finished Training')


# 我们测试数据是CIFAR10，图像大小是 32*32*3
if __name__ == '__main__':
    vgg_train()


