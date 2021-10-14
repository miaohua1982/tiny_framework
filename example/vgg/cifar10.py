import torch as t
from torchvision import transforms
import torchvision
from matplotlib import pyplot as plt

ds_path = '/home/miaohua/Documents/Datasets'  # the datasets path
batch_size = 16
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root=ds_path, train=True, download=False, transform=transform)
trainloader = t.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.CIFAR10(root=ds_path, train=False, download=True, transform=transform)
testloader = t.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

x, y = iter(trainloader).next()

# show one picture
im = x[0,:,:,:]
im = im.permute(1,2,0)  # sent channel to last dimesion
fig = plt.figure()
plotwindow = fig.add_subplot(111)
plt.axis('off')
plt.title(classes[y[0].item()])
plt.imshow(im, cmap='BrBG')
plt.show()