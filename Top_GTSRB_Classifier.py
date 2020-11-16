import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import PIL
from torchvision import transforms
from sklearn.metrics import average_precision_score
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from classifier import SimpleClassifier, Classifier, Classifier_moreConv#, AlexNet
from dataloader import MyDataset, My_classes
from sklearn.model_selection import train_test_split
from pprint import pprint
import sys
import argparse
import time
import simplejson



#========Parameters
num_epochs = 20
test_frequency = 5
batch_size = 64
path ='/GTSRB_data/'
#path = 'G:/My Drive/A-Courses-PhD/Term_17_Fall2020/CS498-DL/assignments/Project/Explainable_GTSRB/GTSRB_data'
num_classes = len(My_classes)

#===========Functions

def train_classifier(train_loader, classifier, criterion, optimizer):
    classifier.train()
    loss_ = 0.0
    losses = []
    for i, (images, labels, _) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = classifier(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss)
    return torch.stack(losses).mean().item()


def test_classifier(test_loader, classifier, criterion, print_ind_classes=True, print_total=True):
    classifier.eval()
    losses = []
    with torch.no_grad():
        y_true = np.zeros((0, num_classes))
        y_score = np.zeros((0, num_classes))
        for i, (images, labels, _) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            logits = classifier(images)
            y_true = np.concatenate((y_true, labels.cpu().numpy()), axis=0)
            y_score = np.concatenate((y_score, logits.cpu().numpy()), axis=0)
            loss = criterion(logits, labels)
            losses.append(loss.item())
        aps = []
        # ignore first class which is background
        for i in range(1, y_true.shape[1]):
            ap = average_precision_score(y_true[:, i], y_score[:, i])
            if print_ind_classes:
                print('-------  Class: {:<12}     AP: {:>8.4f}  -------'.format(My_classes[i], ap))
            aps.append(ap)

        mAP = np.mean(aps)
        test_loss = np.mean(losses)
        if print_total:
            print('mAP: {0:.4f}'.format(mAP))
            print('Avg loss: {}'.format(test_loss))

    return mAP, test_loss, aps


def plot_losses(train, val, test_frequency, num_epochs):
    plt.plot(train, label="train")
    indices = [i for i in range(num_epochs) if ((i + 1) % test_frequency == 0 or i == 0)]
    plt.plot(indices, val, label="val")
    plt.title("Loss Plot")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig('/results/plt/Losses.png')
    plt.show()


def plot_mAP(train, val, test_frequency, num_epochs):
    indices = [i for i in range(num_epochs) if ((i + 1) % test_frequency == 0 or i == 0)]
    plt.plot(indices, train, label="train")
    plt.plot(indices, val, label="val")
    plt.title("mAP Plot")
    plt.ylabel("mAP")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig('/results/plt/MAP.png')
    plt.show()


def train(classifier, num_epochs, train_loader, val_loader, criterion, optimizer, test_frequency=5):
    train_losses = []
    train_mAPs = []
    val_losses = []
    val_mAPs = []
    decayRate = 0.96

    for epoch in range(1, num_epochs + 1):
        print("Starting epoch number " + str(epoch))
        train_loss = train_classifier(train_loader, classifier, criterion, optimizer)
        train_losses.append(train_loss)
        # lr_scheduler.step()
        # print('learning rate :', get_lr(lr_scheduler.optimizer))

        print("Loss for Training on Epoch " + str(epoch) + " is " + str(train_loss))
        if (epoch % test_frequency == 0 or epoch == 1):
            mAP_train, _, _ = test_classifier(train_loader, classifier, criterion, False, False)
            train_mAPs.append(mAP_train)
            mAP_val, val_loss, _ = test_classifier(val_loader, classifier, criterion)
            print('Evaluating classifier')
            print("Mean Precision Score for Testing on Epoch " + str(epoch) + " is " + str(mAP_val))
            val_losses.append(val_loss)
            val_mAPs.append(mAP_val)

    return classifier, train_losses, val_losses, train_mAPs, val_mAPs

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



#======== define image transformation/Augmentation

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std= [0.229, 0.224, 0.225])

train_transform = transforms.Compose([
            #torchvision.transforms.ColorJitter(hue=.1, saturation=.05),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
            transforms.Resize(100),
            transforms.CenterCrop(100),
            transforms.ToTensor(),
            normalize
        ])

test_transform = transforms.Compose([
            transforms.Resize(100),
            transforms.CenterCrop(100),
            transforms.ToTensor(),
            normalize,
        ])

# Create train, validation, and test data set

#TRAIN DATA
s=np.arange(39209)      # total number of training data is 39209
s_train, s_val = train_test_split(s, test_size=0.25, random_state=1)
#s_train, s_test = train_test_split(s_train, test_size=0.25, random_state=1)

ds_train = MyDataset(path,'Train',train_transform, s_train)
ds_val = MyDataset(path,'Train',train_transform, s_val)

# TEST DATA
s=np.arange(12630)      #total number of test data is 12630
#np.random.seed(43)
#np.random.shuffle(s)
ds_test = MyDataset(path,'Test',train_transform, s)
print(ds_train.names.shape)
print(ds_val.names.shape)
print(ds_test.names.shape)




train_loader = torch.utils.data.DataLoader(dataset=ds_train,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=1)

val_loader = torch.utils.data.DataLoader(dataset=ds_val,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=1)

test_loader = torch.utils.data.DataLoader(dataset=ds_test,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=1)


# Running the classifier here
#classifier = SimpleClassifier().to(device)


# Load Pretrained AlexNet
classifier = torchvision.models.alexnet(pretrained=True)
classifier.classifier._modules['6'] = nn.Linear(4096, num_classes)
classifier = classifier.to(device)
optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

criterion = nn.MultiLabelSoftMarginLoss()
#optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, weight_decay = 0.9)
#optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


#decayRate = 0.97
#lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
# optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)

classifier, train_losses, val_losses, train_mAPs, val_mAPs = train(classifier, num_epochs, train_loader, val_loader, criterion, optimizer, test_frequency)

torch.save(classifier.state_dict(), '/results/classifier.pth')
f = open('/results/train_losses.txt', 'w')
simplejson.dump(train_losses, f)
f.close()
f = open('/results/val_losses.txt', 'w')
simplejson.dump(val_losses, f)
f.close()
f = open('/results/train_mAPs.txt', 'w')
simplejson.dump(train_mAPs, f)
f.close()
f = open('/results/val_mAPs.txt', 'w')
simplejson.dump(val_mAPs, f)
f.close()


plot_losses(train_losses, val_losses, test_frequency, num_epochs)
plot_mAP(train_mAPs, val_mAPs, test_frequency, num_epochs)

mAP_test, test_loss, test_aps = test_classifier(test_loader, classifier, criterion)
print(mAP_test)