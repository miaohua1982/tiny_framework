import torch as t
import numpy as np
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


from skimage import transform as sktsf
new_im = x[0,:,:,:]
new_im = sktsf.resize(new_im.detach().cpu().numpy(), (3, 224, 224), mode='reflect', anti_aliasing=False)
new_im = np.transpose(new_im, (1, 2, 0))      # sent channel to last dimesion
print("The new shape is", new_im.shape)
plotwindow = fig.add_subplot(111)
plt.axis('off')
plt.title(classes[y[0].item()])
plt.imshow(im, cmap='BrBG')
plt.show()


new_im = x.clone()
new_im = sktsf.resize(new_im.detach().cpu().numpy(), (16, 3, 224, 224), mode='reflect', anti_aliasing=False)
print("The new shape is", new_im.shape)