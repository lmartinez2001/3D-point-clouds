
#
#
#      0===========================================================0
#      |       TP6 PointNet for point cloud classification         |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 21/02/2023
#

import numpy as np
import random
import math
import os
import time
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import sys

# Import functions to read and write ply files
from ply import write_ply, read_ply



class RandomRotation_z(object):
    def __call__(self, pointcloud):
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),      0],
                               [ math.sin(theta),  math.cos(theta),      0],
                               [0,                               0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud
    
class RandomTranslation(object):
    def __init__(self, magnitude: float = 0.1):
        self.magnitude = magnitude

    def __call__(self, pointcloud):
        translation = np.random.randn(3)
        translation = self.magnitude * translation / np.linalg.norm(translation)
        translated_pointcloud = pointcloud + translation
        return translated_pointcloud
        

        
class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud)


def default_transforms(translation_magnitude: float = 0.0):
    return transforms.Compose([RandomRotation_z(),RandomNoise(),RandomTranslation(magnitude=translation_magnitude), ToTensor()])

def test_transforms():
    return transforms.Compose([ToTensor()])



class PointCloudData_RAM(Dataset):
    def __init__(self, root_dir, folder="train", transform=default_transforms(translation_magnitude=0.1)):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir+"/"+dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.data = []
        for category in self.classes.keys():
            new_dir = root_dir+"/"+category+"/"+folder
            for file in os.listdir(new_dir):
                if file.endswith('.ply'):
                    ply_path = new_dir+"/"+file
                    data = read_ply(ply_path)
                    sample = {}
                    sample['pointcloud'] = np.vstack((data['x'], data['y'], data['z'])).T
                    sample['category'] = self.classes[category]
                    self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pointcloud = self.transforms(self.data[idx]['pointcloud'])
        return {'pointcloud': pointcloud, 'category': self.data[idx]['category']}



class MLP(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(start_dim=1),
                                 self._make_linear(3072, 512),
                                 self._make_linear(512, 256), 
                                 nn.Dropout(p=0.3),
                                 nn.Linear(256, classes))

    def forward(self, input):
        return self.net(input)

    def _make_linear(self, in_features, out_features):
        return nn.Sequential(nn.Linear(in_features, out_features),
                             nn.BatchNorm1d(out_features),
                             nn.ReLU())



class PointNetBasic(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.shared1 = nn.Sequential(self._make_linear(3, 64, shared=True),
                                     self._make_linear(64, 64, shared=True))
        
        self.shared2 = nn.Sequential(self._make_linear(64, 64, shared=True),
                                     self._make_linear(64, 128, shared=True),
                                     self._make_linear(128, 1024, shared=True))
        
        self.mlp = nn.Sequential(self._make_linear(1024, 512, shared=False),
                                 self._make_linear(512, 256, shared=False),
                                 nn.Dropout(p=0.3),
                                 nn.Linear(256, classes))
        
        self.maxpool = nn.MaxPool1d(kernel_size=1024)

    def forward(self, input):
        # input (B, 3, 1024)
        x = self.shared1(input) # (B, 64, 1024)
        x = self.shared2(x) # (B, 1024, 1024)
        x = self.maxpool(x).squeeze(-1) # (B, 1024)
        x = self.mlp(x) # (B, 10)
        return x

    
    def _make_linear(self, in_features, out_features, shared=False):
        layers = []
        if shared:
            layers += [nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1)]
        else:
            layers += [nn.Linear(in_features, out_features)]
        layers += [nn.BatchNorm1d(out_features), nn.ReLU()]
        return nn.Sequential(*layers)
        

class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.shared = nn.Sequential(self._make_linear(3, 64, shared=True),
                                    self._make_linear(64, 128, shared=True),
                                    self._make_linear(128, 1024, shared=True))
        
        self.maxpool = nn.MaxPool1d(kernel_size=1024)

        self.mlp = nn.Sequential(self._make_linear(1024, 512, shared=False),
                                 self._make_linear(512, 256, shared=False),
                                 nn.Linear(256, k**2))

    def forward(self, input):
        b, _, _ = input.shape
        x = self.shared(input)
        x = self.maxpool(x).squeeze(-1)
        x = self.mlp(x)
        x = x.view(b, self.k, self.k)
        return x + torch.eye(self.k, device=input.device).unsqueeze(0)

    def _make_linear(self, in_features, out_features, shared=False):
        layers = []
        if shared:
            layers += [nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1)]
        else:
            layers += [nn.Linear(in_features, out_features)]
        layers += [nn.BatchNorm1d(out_features), nn.ReLU()]
        return nn.Sequential(*layers)


class PointNetFull(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.input_transform = Tnet(k=3)

        self.shared1 = nn.Sequential(self._make_linear(3, 64, shared=True),
                                     self._make_linear(64, 64, shared=True))
        
        self.shared2 = nn.Sequential(self._make_linear(64, 64, shared=True),
                                     self._make_linear(64, 128, shared=True),
                                     self._make_linear(128, 1024, shared=True))
        
        self.mlp = nn.Sequential(self._make_linear(1024, 512, shared=False),
                                 self._make_linear(512, 256, shared=False),
                                 nn.Dropout(p=0.3),
                                 nn.Linear(256, classes))
        
        self.maxpool = nn.MaxPool1d(kernel_size=1024)

    def forward(self, input):        
        in_transform = self.input_transform(input)
        x = torch.einsum('bij,bik->bij', input, in_transform)
        x = self.shared1(x)
        x = self.shared2(x)
        x = self.maxpool(x).squeeze(-1)
        x = self.mlp(x)
        return x, in_transform
    
    def _make_linear(self, in_features, out_features, shared=False):
        layers = []
        if shared:
            layers += [nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1)]
        else:
            layers += [nn.Linear(in_features, out_features)]
        layers += [nn.BatchNorm1d(out_features), nn.ReLU()]
        return nn.Sequential(*layers)



def basic_loss(outputs, labels):
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(outputs, labels)

def pointnet_full_loss(outputs, labels, m3x3, alpha = 0.001):
    criterion = torch.nn.CrossEntropyLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.to(m3x3.device)
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)) / float(bs)



def train(model, device, train_loader, test_loader=None, epochs=250):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss=0
    best_acc = 0
    for epoch in range(epochs): 
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            # outputs = model(inputs.transpose(1,2))
            outputs, m3x3 = model(inputs.transpose(1,2)) # (32, 3, 1024)
            # loss = basic_loss(outputs, labels)
            loss = pointnet_full_loss(outputs, labels, m3x3)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        test_acc = 0
        if test_loader:
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    # outputs = model(inputs.transpose(1,2))
                    outputs, __ = model(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            test_acc = 100. * correct / total
            if test_acc > best_acc: best_acc = test_acc
            print('Epoch: %d, Loss: %.3f, Test accuracy: %.1f %%' %(epoch+1, loss, test_acc))
    print(f"Best test accuracy: {best_acc:.1f}")


 
if __name__ == '__main__':
    
    t0 = time.time()
    
    # ROOT_DIR = "../data/ModelNet10_PLY"
    ROOT_DIR = "../data/ModelNet40_PLY"
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    
    train_ds = PointCloudData_RAM(ROOT_DIR, folder='train', transform=default_transforms())
    test_ds = PointCloudData_RAM(ROOT_DIR, folder='test', transform=test_transforms())

    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    print("Classes: ", inv_classes)
    print('Train dataset size: ', len(train_ds))
    print('Test dataset size: ', len(test_ds))
    print('Number of classes: ', len(train_ds.classes))
    print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())

    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=32)

    # model = MLP(classes=len(train_ds.classes))
    # model = PointNetBasic(classes=len(train_ds.classes))
    model = PointNetFull(classes=len(train_ds.classes))
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print("Number of parameters in the Neural Networks: ", sum([np.prod(p.size()) for p in model_parameters]))
    model.to(device);
    
    train(model, device, train_loader, test_loader, epochs = 250)
    
    t1 = time.time()
    print("Total time for training : ", t1-t0)

    
    


