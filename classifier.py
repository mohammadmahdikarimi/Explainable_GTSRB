import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import math

from dataloader import My_classes
# NUM_CLASSES = 55
NUM_CLASSES = len(My_classes)

class SimpleClassifier(nn.Module):
    def __init__(self):
        #self.NUM_CLASSES = NUM_CLASSES
        #super() allows you to build classes that easily extend the functionality of previously built classes without implementing their functionality again.
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size()[0], 16 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Alexnet_Classifier(nn.Module):
    def __init__(self):
        #super() allows you to build classes that easily extend the functionality of previously built classes without implementing their functionality again.
        super(Alexnet_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.conv2 = nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv3 = nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.fc1 = nn.Linear(in_features=9216, out_features=4096, bias=True)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.fc3 = nn.Linear(in_features=4096, out_features=NUM_CLASSES, bias=True)
        self.drop = nn.Dropout(p=0.5, inplace=False)
        print("Alex")


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        print("t1_Alex: ",x.size())
        x = x.view(x.size()[0], 9216)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Azin_Classifier(nn.Module):
    def __init__(self):
        #super() allows you to build classes that easily extend the functionality of previously built classes without implementing their functionality again.
        super(Azin_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(7, 7), stride=(1, 1), padding=(1, 1)) #74
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1)) #36
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)) #19
        
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=1, ceil_mode=False)
        self.fc1 = nn.Linear(in_features=  64 * 26 * 26, out_features=64 * 26, bias=True)
        self.fc2 = nn.Linear(in_features=64 * 26, out_features=64, bias=True)
        self.fc3 = nn.Linear(in_features=64 , out_features=NUM_CLASSES, bias=True)
        
        self.drop = nn.Dropout(p=0.5, inplace=False)
        print('My_Classifier')


    def forward(self, x):
        #print("t1_n: ",x.size())
        #
        x = F.relu(self.conv1(x)) #223
        #print("t2_: ",x.size())

        x = self.pool2(x) # --> 111
        #print("t3: ",x.size())
        #res = x        

        x = F.relu(self.conv2(x)) # --> 109
        #print("t4: ",x.size())

        x = self.pool2(x) # --> 54
        #print("t5: ",x.size())
        
        x = F.relu(self.conv3(x)) # 56 - 3 + 2 + 1 --> 56 
        #print("t6: ",x.size())

        x = self.pool2(x) # --> 28
        #x += res
        #print("t7: ",x.size())

        
        #x = self.pool2(F.relu(self.conv4(x)))
        #x += res
        #plt.imshow(x[0].cpu().numpy())
        #print("t5: ",x.size())
        
        #x = F.relu(self.conv3(x))
        #print("t6: ",x.size())
        
        #x = F.relu(self.conv5(x))
        #print("t7: ",x.size())
        
        #x = F.relu(self.conv6(x))
        #print("t8: ",x.size())
        
        #x = F.relu(self.conv4(x))
        #print("t9: ",x.size())
        x = x.view(x.size()[0], 64 * 26 * 26) #flatten to 50176
        #x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x




class Classifier(nn.Module):
    def __init__(self):
        #super() allows you to build classes that easily extend the functionality of previously built classes without implementing their functionality again.
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(3, 3), padding=(2, 2)) #74
        self.conv2 = nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)) #36
        self.conv3 = nn.Conv2d(192, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) #19
        self.conv4 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) #10
        
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
        self.fc1 = nn.Linear(in_features= 2048, out_features=2048, bias=True)
        self.fc2 = nn.Linear(in_features=2048, out_features=200, bias=True)
        self.fc3 = nn.Linear(in_features=200, out_features=NUM_CLASSES, bias=True)
        self.drop = nn.Dropout(p=0.5, inplace=False)
        print('My_Classifier')


    def forward(self, x):
        #print("t1_n: ",x.size())
        #
        x = F.relu(self.conv1(x))
        #print("t2: ",x.size())
        
        x = self.pool2(F.relu(self.conv2(x)))
        #print("t3: ",x.size())
        
        res = x        
        x = self.pool2(F.relu(self.conv3(x)))
        #print("t4: ",x.size())
        
        x = self.pool2(F.relu(self.conv4(x)))
        #x += res
        #plt.imshow(x[0].cpu().numpy())
        #print("t5: ",x.size())
        
        #x = F.relu(self.conv3(x))
        #print("t6: ",x.size())
        
        #x = F.relu(self.conv5(x))
        #print("t7: ",x.size())
        
        #x = F.relu(self.conv6(x))
        #print("t8: ",x.size())
        
        #x = F.relu(self.conv4(x))
        #print("t9: ",x.size())

        x = x.view(x.size()[0], 2048)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        



class Classifier_moreConv(nn.Module):
    def __init__(self):
        #super() allows you to build classes that easily extend the functionality of previously built classes without implementing their functionality again.
        super(Classifier_moreConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(3, 3), padding=(2, 2)) #74
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)) #36
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) #19
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) #10
        
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
        
        self.ad_pool2 = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        
        
        self.fc1 = nn.Linear(in_features= 25088, out_features=1024, bias=True)
        self.fc2 = nn.Linear(in_features=1024, out_features=200, bias=True)
        self.fc3 = nn.Linear(in_features=200, out_features=NUM_CLASSES, bias=True)
        self.drop = nn.Dropout(p=0.3, inplace=False)
        print('My_Classifier')


    def forward(self, x):
        #print("t1_n: ",x.size())
        #
        x = F.relu(self.conv1(x))
        #print("t2: ",x.size())
        
        x = self.pool2(F.relu(self.conv2(x)))
        #print("t3: ",x.size())
        
        res = x        
        x = self.pool2(F.relu(self.conv3(x)))
        #print("t4: ",x.size())
        
        x = self.pool2(F.relu(self.conv4(x)))
        #x += res
        #plt.imshow(x[0].cpu().numpy())
        #print("t5: ",x.size())
        
        x = self.ad_pool2(x)
        
        #print("t5_1: ",x.size())
        
        x = x.view(x.size()[0], 25088)
        
        #print("t6: ",x.size())
        
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        
        #print("t7: ",x.size())
        
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        
        #print("t8: ",x.size())
        
        x = self.fc3(x)
        return x
        
        



     
        # add VGG model 
class Classifier_11(nn.Module):
    def __init__(self):
        #super() allows you to build classes that easily extend the functionality of previously built classes without implementing their functionality again.
        super(Classifier_11, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(3, 3), padding=(2, 2)) #74
        self.conv2 = nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)) #36
        self.conv3 = nn.Conv2d(192, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) #19
        
        #continue increase feature map then decrease the size of kernel_size
        #add res , VGG
        #adaptive average pooling
        #have more of 3x3 conv layer
        
        self.conv4 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) #10
        
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
        self.fc1 = nn.Linear(in_features= 2048, out_features=1024, bias=True)
        self.fc2 = nn.Linear(in_features=1024, out_features=200, bias=True)
        self.fc3 = nn.Linear(in_features=200, out_features=NUM_CLASSES, bias=True)
        self.drop = nn.Dropout(p=0.5, inplace=False)
        print('My_Classifier')


    def forward(self, x):
        #print("t1_n: ",x.size())
        #
        x = F.relu(self.conv1(x))
        #print("t2: ",x.size())
        
        x = self.pool2(F.relu(self.conv2(x)))
        #print("t3: ",x.size())
        
        res = x        
        x = self.pool2(F.relu(self.conv3(x)))
        #print("t4: ",x.size())
        
        x = self.pool2(F.relu(self.conv4(x)))
        #x += res
        #plt.imshow(x[0].cpu().numpy())
        #print("t5: ",x.size())
        
        #x = F.relu(self.conv3(x))
        #print("t6: ",x.size())
        
        #x = F.relu(self.conv5(x))
        #print("t7: ",x.size())
        
        #x = F.relu(self.conv6(x))
        #print("t8: ",x.size())
        
        #x = F.relu(self.conv4(x))
        #print("t9: ",x.size())

        x = x.view(x.size()[0], 2048)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class vgg_Classifier(nn.Module):
    def __init__(self):
        #super() allows you to build classes that easily extend the functionality of previously built classes without implementing their functionality again.
        super(vgg_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #74
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #36
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #19
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #10
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #10
        self.conv6 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #10
        self.conv7 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #10
        self.conv8 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #10
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.ad_pool2 = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        
        
        self.fc1 = nn.Linear(in_features= 25088, out_features=4096, bias=True)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.fc3 = nn.Linear(in_features=4096, out_features=NUM_CLASSES, bias=True)
        self.drop = nn.Dropout(p=0.5, inplace=False)
        print('My_Classifier')


    def forward(self, x):

        x = self.pool2(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool2(F.relu(self.conv4(x)))
        x = self.pool2(F.relu(self.conv5(x)))
        x = self.pool2(F.relu(self.conv6(x)))
        x = self.pool2(F.relu(self.conv7(x)))
        
        x = self.ad_pool2(x)
        
        a = x.size()[1]*x.size()[2]*x.size()[3]
        #print('size is:',a)
        #print('size of x:', x.size())
        x = x.view(x.size()[0], a)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        
        return x
        
        
class ConvCat_Classifier(nn.Module):
    def __init__(self):
        #super() allows you to build classes that easily extend the functionality of previously built classes without implementing their functionality again.
        super(ConvCat_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(3, 3), padding=(2, 2)) #74
        self.conv2 = nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)) #36
        self.conv3 = nn.Conv2d(192, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) #19
        self.conv4 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) #10
        
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
        
        
        self.fc1 = nn.Linear(in_features= 267328, out_features=33416, bias=True)
        self.fc2 = nn.Linear(in_features=33416, out_features=2048, bias=True)
        self.fc3 = nn.Linear(in_features=2048, out_features=2048, bias=True)
        self.fc4 = nn.Linear(in_features=2048, out_features=200, bias=True)
        self.fc5 = nn.Linear(in_features=200, out_features=NUM_CLASSES, bias=True)
        self.drop = nn.Dropout(p=0.5, inplace=False)
        print('My_Classifier')


    def forward(self, x):
        #print("t1_n: ",x.size())
        #
        x = F.relu(self.conv1(x))
        #print("t2: ",x.size())
        #x1 = x
        x = F.relu(self.conv2(x))
        #print("t3: ",x.size())
        #x2 = x
        a2 = x.size()[1]*x.size()[2]*x.size()[3]
        #print('data 2 size:', a2)

        
        x = F.relu(self.conv3(x))
        #print("t4: ",x.size())
        x3 = x
        a3 = x.size()[1]*x.size()[2]*x.size()[3]
        
        x = F.relu(self.conv4(x))
        a4 = x.size()[1]*x.size()[2]*x.size()[3]
       
        
        
        #print('data 3 size:', a3)
   
        
        #x = torch.cat((x,x3),0)
        
        x = x.view(x.size()[0], a3)
        print(x.size())

        x = F.relu(self.fc1(x))
        x = self.drop(x)
        
        x = F.relu(self.fc2(x))
        #x = self.drop(x)
        
        #x = F.relu(self.fc3(x))
        #x = self.drop(x)
        
        x = F.relu(self.fc4(x))
        x = self.drop(x)
        
        x = self.fc5(x)
        return x

class ConvLayer_Classifier(nn.Module):
    def __init__(self):
        #super() allows you to build classes that easily extend the functionality of previously built classes without implementing their functionality again.
        super(ConvLayer_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(3, 3), padding=(2, 2)) #74
        self.conv2 = nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)) #36
        self.conv3 = nn.Conv2d(192, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) #19
        self.conv4 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) #10
        
        self.convnet1 = nn.Conv2d(3, 64, kernel_size= (1,1), strides = (2,2))
        self.convnet2 = nn.Conv2d(64, 64, kernel_size= (3, 3), strides = (1,1))
        self.convnet3 = nn.Conv2d(64, 256, kernel_size= (1, 1), strides = (1,1))
        
        self.convnetshort1 = nn.Conv2d(3, 256, kernel_size= (1, 1), strides = (1,1))
        
        self.BatchNorm = nn.BatchNorm2d(axis = 3)
        
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
        
        
        self.fc1 = nn.Linear(in_features= 267328, out_features=33416, bias=True)
        self.fc2 = nn.Linear(in_features=33416, out_features=2048, bias=True)
        self.fc3 = nn.Linear(in_features=2048, out_features=2048, bias=True)
        self.fc4 = nn.Linear(in_features=2048, out_features=200, bias=True)
        self.fc5 = nn.Linear(in_features=200, out_features=NUM_CLASSES, bias=True)
        self.drop = nn.Dropout(p=0.5, inplace=False)
        print('My_Classifier')


    def forward(self, x):
        x_res = x
        #
        x = F.relu(self.BatchNorm(self.convnet1(x)))
        x = F.relu(self.BatchNorm(self.convnet2(x)))
        x = F.relu(self.BatchNorm(self.convnet3(x)))
        x_res = F.relu(self.BatchNorm(self.convnetshort1(x_res)))
        x = torch.cat((x,x_res),0)
        x = F.relu(x)
        
        print(x.size())

        x = F.relu(self.fc1(x))
        x = self.drop(x)
        
        x = F.relu(self.fc2(x))
        #x = self.drop(x)
        
        #x = F.relu(self.fc3(x))
        #x = self.drop(x)
        
        x = F.relu(self.fc4(x))
        x = self.drop(x)
        
        x = self.fc5(x)
        return x


class Classifier_maz(nn.Module):
    def __init__(self):
        #super() allows you to build classes that easily extend the functionality of previously built classes without implementing their functionality again.
        super(Classifier_maz, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(1, 1), padding=(2, 2)) #74
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)) #36
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #19
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #10
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #10
        self.conv6 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #10
        self.conv7 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #10
        self.conv8 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #10
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.ad_pool2 = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        

        self.fc1 = nn.Linear(in_features= 25088, out_features=2048, bias=True)
        self.fc2 = nn.Linear(in_features=2048, out_features=200, bias=True)
        self.fc3 = nn.Linear(in_features=200, out_features=NUM_CLASSES, bias=True)
        self.drop = nn.Dropout(p=0.2, inplace=False)
        print('My_Classifier')


    def forward(self, x):
        
        #t1_new:  torch.Size([64, 3, 227, 227])
        #t2:  torch.Size([64, 64, 110, 110])
        #t3:  torch.Size([64, 128, 55, 55])
        #t4:  torch.Size([64, 256, 27, 27])
        #t5:  torch.Size([64, 512, 13, 13])
        #t6:  torch.Size([64, 512, 6, 6])
        #t7:  torch.Size([64, 512, 3, 3])
        #t8:  torch.Size([64, 512, 3, 3])
        #t9:  torch.Size([64, 512, 3, 3])
        #t9_1:  torch.Size([64, 512, 7, 7])
        #t10:  torch.Size([64, 25088])
        #t11:  torch.Size([64, 2048])
        #t12:  torch.Size([64, 200])
        
        
        #print("t1_new: ",x.size())

        x = self.pool2(F.relu(self.conv1(x)))
        #print("t2: ",x.size())
        
        x = self.pool2(F.relu(self.conv2(x)))
        #print("t3: ",x.size())
             
        x = self.pool2(F.relu(self.conv3(x)))
        #print("t4: ",x.size())
        
        x = self.pool2(F.relu(self.conv4(x)))
        #print("t5: ",x.size())
        
        x = self.pool2(F.relu(self.conv5(x)))
        #print("t6: ",x.size())    
        
        x = self.pool2(F.relu(self.conv6(x)))
        #print("t7: ",x.size())    
        
        x = F.relu(self.conv7(x))
        #print("t8: ",x.size())       
        
        x = F.relu(self.conv8(x))
        #print("t9: ",x.size())
        
        x = self.ad_pool2(x)
        #print("t9_1: ",x.size())
        
        #x = F.relu(self.conv3(x))
        #print("t6: ",x.size())
        
        #x = F.relu(self.conv5(x))
        #print("t7: ",x.size())
        
        #x = F.relu(self.conv6(x))
        #print("t8: ",x.size())
        
        #x = F.relu(self.conv4(x))
        #print("t9: ",x.size())

        x = x.view(x.size()[0], 25088)
        
        #print("t10: ",x.size())
        
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        
        #print("t11: ",x.size())
        
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        
        #print("t12: ",x.size())
        
        x = self.fc3(x)
        return x