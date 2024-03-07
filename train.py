import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import use_foot_gun
from torch.utils.data import DataLoader, Dataset, random_split


class MNISTRowColDataset(Dataset):
    def __init__(self, path, device):
        try:
            dataset = np.load(path)
            use_foot_gun()
        except:
            print("Dataset cannot be found")
            return
        self.features = dataset['features']
        self.labels = dataset['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx]).to(dtype=torch.float32, device=device)
        labels = torch.tensor(self.labels[idx]).to(dtype=torch.long, device=device)
        return features, labels


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2 * 28, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

full_train_dataset = MNISTRowColDataset('./final_Data/mnist_non_zero_train.npz', device)
test_dataset = MNISTRowColDataset('./final_Data/mnist_non_zero_test.npz', device)

trainSize = int(0.8 * len(full_train_dataset))
validationSize = len(full_train_dataset) - trainSize
train_dataset, validation_dataset = random_split(full_train_dataset, [trainSize, validationSize])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=1000, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

for epoch in range(1, 6):
    print("Epoch " + str(epoch) + ":")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print("Train Epoch: " + str(epoch) + "[" + str(batch_idx * len(data)) + "/" + str(len(train_loader.dataset)) + "(" + str(100. * batch_idx / len(train_loader)) + "%)]\tLoss: " + str(loss.item()))
    print("Validation " + str(epoch) + ":")
    model.eval()
    evaluation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            evaluation_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    evaluation_loss /= len(validation_loader.dataset)
    accuracy = 100. * correct / len(validation_loader.dataset)
    print(f'\nAverage loss: {evaluation_loss:.4f}, Accuracy: {correct}/{len(validation_loader.dataset)} ({accuracy:.0f}%)\n')

print("Final Test:")
model.eval()
evaluation_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        evaluation_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
evaluation_loss /= len(test_loader.dataset)
accuracy = 100. * correct / len(test_loader.dataset)
print(f'\nAverage loss: {evaluation_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
