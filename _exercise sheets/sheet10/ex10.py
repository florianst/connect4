import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from datetime import datetime

print('Execution started: ')
print(datetime.now().time())

# load CIFAR10
transform = transforms.Compose([transforms.Scale(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=1)
trainloader_new = trainloader

#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# define loss function and optimizer
import torch.optim as optim
import torch.nn as nn
criterion = nn.CrossEntropyLoss()

file = open('results.txt', 'w')

# train the network
from torch.autograd import Variable
for learning_rate in (1e-6, 1e-5, 5e-5, 0.0001, 0.001, 0.005):
    # set up VGG16 network architecture
    network = models.vgg16(pretrained=False)
    network.cuda() # enables GPU usage

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    file.write('lr=%f\r\n' % learning_rate)

    for epoch in range(5):  # loop over the dataset five times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda()) # CPU tensors in inputs, labels get converted to GPU ones here

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 1000 == 999:    # print every 1000 mini-batches
                print('%d, %d, %f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                file.write('%d, %d, %f \r\n' %(epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

    trainloader = trainloader_new # load unoptimised training set
    file.write('\r\n')


file.close()

print('Execution finished: ')
print(datetime.now().time())