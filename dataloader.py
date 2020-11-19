import sys
#import random
import os, numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
#from skiage.transform import resize
#from scipy.sparse import csr_matrix
from PIL import Image
import xml.etree.ElementTree as ET
import csv

#import cv2

#import matplotlib.pyplot as plt


My_classes = []
Extended_cls = ('red', 'blue', 'black', 'circle', 'triangle',
                  'blk cross line', 'number 8','number 2', 'number 1', 'number 0',
                  'two cars', 'car and truck')
for i in range(43):
    My_classes.append(str(i))
for i in Extended_cls:
    My_classes.append(i)

class MyDataset(data.Dataset):
    def __init__(self, data_path, dataset_split, transform, file_Indx, random_crops=0):
        self.data_path = data_path
        self.transform = transform   #data transformation, augmentation  #from torchvision import transforms
        self.random_crops = random_crops
        self.dataset_split = dataset_split  #train, val, test
        self.num_trafficsigns = 43
        self.file_Indx = file_Indx      #train/test/val are indexed from total image pool given here one by one

        self.__init_classes()
        self.names, self.labels, self.box_indices, self.label_order = self.__dataset_info()

    def __getitem__(self, index):
        # CHANGED
#         x = imread( self.names[index], mode='RGB')
#         x = Image.fromarray(x)
        x = Image.open(self.names[index])

        scale = np.random.rand() * 2 + 0.25
        w = int(x.size[0] * scale)
        h = int(x.size[1] * scale)
        if min(w, h) < 227:
            scale = 227 / min(w, h)
            w = int(x.size[0] * scale)
            h = int(x.size[1] * scale)

        if self.random_crops == 0:
            x = self.transform(x)
        else:
            crops = []
            for i in range(self.random_crops):
                crops.append(self.transform(x))
            x = torch.stack(crops)

        y = self.labels[index]
        z = self.box_indices[index]
        return x, y, z

    def __len__(self):
        return len(self.names)

    def __init_classes(self):
        self.classes = My_classes
        self.num_classes = len(self.classes)  # 43 + len(additional classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))

    def ex_cls(self):
        class_map = dict()
        for i in range(43):
            class_map[str(i)] = []
        with open(self.data_path + '/ImageSets/Main/ex_cls.csv') as csv_file:
            csv_reader = csv.reader(csv_file)
            for i, line in enumerate(csv_reader):
                if i !=0:
                    cls = [i for i, x in enumerate(line[1:]) if x == "1"]
                    class_map[str(line[0])] = cls
        return class_map



    def __dataset_info(self):

        img = dict()
        img['name'] = []
        img['box'] = []     #x1,,y1,x2,y2
        img['cls_id'] = []
        img['size'] = []    #width,height
        img['label_order'] = []
        img['labels'] = []
        class_map = self.ex_cls()
        with open(self.data_path + '/ImageSets/Main/' + self.dataset_split + '.csv') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for line in csv_reader:
                img['cls_id'].append(line['ClassId'])
                img['size'].append([line['Width'], line['Height']])
                img['name'].append(os.path.join(self.data_path, line['Path']))

                # Make pixel indexes 0-based
                x1 = float(line['Roi.X1']) - 1
                y1 = float(line['Roi.Y1']) - 1
                x2 = float(line['Roi.X2']) - 1
                y2 = float(line['Roi.Y2']) - 1
                img['box'].append([x1, y1, x2, y2])

                img['label_order'].append(class_map[line['ClassId']])

                lbl = np.zeros(self.num_classes)
                lbl[class_map[line['ClassId']]] = 1
                img['labels'].append(lbl)
        dum = 0
        return np.array(img['name'])[self.file_Indx], np.array(img['labels']).astype(np.float32)[self.file_Indx], np.array(img['box'])[self.file_Indx], np.array(img['label_order'])[self.file_Indx]


if __name__ == "__main__":
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std= [0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
                transforms.Resize(227),
                transforms.CenterCrop(227),
                transforms.ToTensor(),
                normalize
            ])
    path = os.getcwd()
    # Randomize the order of the input images
    s = np.arange(39209)
    np.random.seed(43)
    np.random.shuffle(s)
    ds_train = MyDataset(path + '/GTSRB_data/' ,'Train',train_transform, s)
    print('this')