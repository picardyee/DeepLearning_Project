import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt

# 데이터 준비

# 데이터 포멧설정
mnist_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.5,), std=(1,))])

# 데이터 다운로드 or 불러오기
trainset = datasets.MNIST(root='/Users/picardy/PycharmProjects/DeepLearning_Project/',
                          train=True, download=True,
                          transform = mnist_transform)
testset = datasets.MNIST(root='/Users/picardy/PycharmProjects/DeepLearning_Project/',
                          train=False, download=True,
                          transform = mnist_transform)

# 데이터 로드 설정
train_loader = DataLoader(trainset, batch_size=8, shuffle=True)
test_loader = DataLoader(testset, batch_size=8, shuffle=False)

# 데이터 로드 확인
dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images.shape)
print(labels.shape)
torch_image = torch.squeeze(images[0])
print(torch_image.shape)

# 시각화
figure = plt.figure(figsize=(12,6))
cols, rows = 4, 2
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(trainset), size=(1,)).item()
    img, label = trainset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis('off')
    plt.imshow(torch.squeeze(img), cmap='gray')
plt.show()


# 신경망 구성