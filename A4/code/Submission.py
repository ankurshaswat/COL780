#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nnet import Net
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import f1_score
from utils import TRANSFORM, NUM_CHANNELS, imshow
import torchvision
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

EPOCHS = 100
BATCH_SIZE = 1024
LR = 0.1
MOMENTUM = 0.9
plot = False
TRAIN = True

def load_dataset_from_folder(all_data_path='./../data/Simple/', validation_split_size=0.1, batch_size=16, num_workers=6, shuffle=True):
    all_data = ImageFolder(
        root=all_data_path,
        transform=TRANSFORM
    )

    classes = all_data.classes

    validation_size = int(validation_split_size * len(all_data))
    test_size = int(validation_split_size * len(all_data))
    train_size = len(all_data) - 2*validation_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        all_data, [train_size, validation_size, test_size])

    training_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )

    validation_dataset_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )

    test_dataset_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
    return training_data_loader, validation_dataset_loader, test_dataset_loader, classes


# In[4]:

net = Net(NUM_CHANNELS)
try:
    net  = net.load("../models/1573584871.3391135_12.model")
except Exception as e:
    print("NNET NOT FOUND: ", str(e))
    pass

CUDA = False
if torch.cuda.is_available():
    CUDA = True
    print('Cuda found')
    device = torch.cuda.current_device()
    print(device)
    net = net.cuda()

trainloader, valloader, testloader, classes = load_dataset_from_folder(
    batch_size=BATCH_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)
optimizer = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images[:,:3,:,:]))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# # show images
# imshow(torchvision.utils.make_grid(images[:,3:4,:,:]))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# # show images
# imshow(torchvision.utils.make_grid(images[:,4:5,:,:]))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

with open ('log_val', 'a') as fv:
    with open('log_lab', 'a') as lab:
        with open('log_train', 'a') as f:
            for epoch in range(EPOCHS):
                if TRAIN:
                    running_loss = 0.0
                    num_batches = 0
                    for i, data in enumerate(trainloader):
                        inputs, labels = data
                        if CUDA:
                            inputs = inputs.cuda()
                            labels = labels.cuda()
                        
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                        num_batches += 1
                        if i % 1 == 0:
                            print('[%d, %5d] loss: %.3f' %
                                  (epoch + 1, i + 1, running_loss / 10))
                            f.write(str(running_loss/10)+"\n")
                            running_loss = 0.0

                    name = net.save(classes, epoch)
                    print('Model saved as {}'.format(name))
                    y_true = []
                    y_pred = []
                    class_correct = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))

                    with torch.no_grad():
                        for data in valloader:
                            images, labels = data
                            labels_cp = labels.clone()
                            if CUDA:
                                images = images.cuda()
                                labels = labels.cuda()
                            outputs = net(images)
                            loss = criterion(outputs, labels)
                            _, predicted = torch.max(outputs, 1)
                            predicted = predicted.cpu()
                            for i in range(len(predicted)):
                                y_pred.append(predicted[i])
                                y_true.append(labels_cp[i])
                            c = (predicted == labels_cp).squeeze()
                            for i in range(min(BATCH_SIZE, len(labels_cp))):
                                label = labels_cp[i]
                                class_correct[label] += c[i].item()
                                class_total[label] += 1
                            fv.write(str(loss.item())+"\n")
                        lab.write(str(len(class_correct)/len(class_total))+"\n")
                        print("Validation: F1: ", f1_score(y_true, y_pred, average='weighted'))
                y_true = []
                y_pred = []
                class_correct = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data
                        labels_cp = labels.clone()
                        if CUDA:
                            images = images.cuda()
                            labels = labels.cuda()
                        outputs = net(images)
                        loss = criterion(outputs, labels)
                        _, predicted = torch.max(outputs, 1)
                        predicted = predicted.cpu()
                        for i in range(len(predicted)):
                            y_pred.append(predicted[i])
                            y_true.append(labels_cp[i])
                        c = (predicted == labels_cp).squeeze()
                        for i in range(min(BATCH_SIZE, len(labels_cp))):
                            label = labels_cp[i]
                            class_correct[label] += c[i].item()
                            class_total[label] += 1
                print("Test F1: ", f1_score(y_true, y_pred, average='weighted'))
                if plot:
                    plot_confusion_matrix(y_true, y_pred, classes=class_names,title='Confusion matrix, without normalization')
                    plt.show()
