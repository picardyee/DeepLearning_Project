import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torchvision


# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# 데이터 로드
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, ), (0.5, ))])

trainset = datasets.FashionMNIST(root='/Users/picardy/PycharmProjects/DeepLearning_Project/',
                                 train = True, download=True,
                                 transform=transform)
testset = datasets.FashionMNIST(root='/Users/picardy/PycharmProjects/DeepLearning_Project/',
                                 train = False, download=True,
                                 transform=transform)

train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
test_loader = DataLoader(testset, batch_size=128, shuffle=False)

images, labels = next(iter(train_loader))
# print(images.shape)
# print(labels.shape)

labels_map = {
    0: 'T-Shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot',
}

figure = plt.figure(figsize=(12, 12))
cols, rows = 4, 4
for i in range(1, cols * rows + 1):
    image = images[i].squeeze()
    label_idx = labels[i].item()
    label = labels_map[label_idx]

    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis('off')
    plt.imshow(image, cmap='gray')
# plt.show()


# 모델 정의 및 파라미터
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features

net = NeuralNet()
# print(net)

params = list(net.parameters())

print(len(params))
print(params[0].size())


# 손실함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 모델 학습
total_batch = len(train_loader)
print('total : {}'.format(total_batch))

# for epoch in range(10):
#
#     running_loss = 0.0
#
#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data
#
#         optimizer.zero_grad()
#
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#         if i % 100 == 99:
#             print('Epoch: {}, Iter: {}, Loss: {}'.format(epoch+1, i+1, running_loss / 2000))
#             running_loss = 0.0
#
#
# # 모델의 저장 및 로드
PATH = './FashionMNIST.pth'
# torch.save(net.state_dict(), PATH)

net = NeuralNet()
net.load_state_dict(torch.load(PATH))

print(net.parameters)


# 모델 테스트
def imshow(image):
    image = image / 2 + 0.5
    npimg = image.numpy()

    fig = plt.figure(figsize=(12, 6))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(test_loader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images[:6]))

outputs = net(images)

_, pred = torch.max(outputs, 1)
print(pred)

print(''.join('{} '.format(labels_map[int(pred[j].numpy())]) for j in range(6)))

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(correct * 100 / total)