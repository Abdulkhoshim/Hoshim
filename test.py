from __future__ import print_function

import scipy
from scipy import io
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from network_4 import Net
import os
import time
import paras
import xlwt

net = Net().cuda()

# GPU
USE_GPU = paras.USE_GPU

# image parameters
h, w, c = paras.h, paras.w, paras.c
sub_image_size = paras.sub_image_size
stride = paras.stride
rows = int((h - sub_image_size) / stride + 1)
columns = int((w - sub_image_size) / stride + 1)

# hyper-parameters
lr = paras.lr
momentum = paras.momentum
epoch = paras.epoch
batch_size = paras.batch_size

# data load for crack_43
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
test_set = ImageFolderWithPaths(paras.test_image_12, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(Net().parameters(), lr=paras.lr)

# test
#net = Net
net = torch.load('model/network_4/image_86_small/model_20.pkl')
net.eval()


number_of_sub_images_in_folder = len(test_loader.dataset)
# number_of_sub_images_in_folder = len([f for f in os.listdir('test_image/test_05/class0') if os.path.isfile(os.path.join('test_image/test_05/class0', f))])

# number_of_sub_images_in_folder = len([f for f in os.listdir('test_image/crack_43/0_background') if os.path.isfile(os.path.join('test_image/crack_43/0_background', f))])+ len([f for f in os.listdir('test_image/crack_43/1_crack') if
#                                       os.path.isfile(os.path.join('test_image/crack_43/1_crack', f))])
print(number_of_sub_images_in_folder)
a = 0
b = paras.batch_size
itr = (number_of_sub_images_in_folder / b)
print(itr)
print('Testing Process:________')

classes = ['background', 'crack']
confusion_matrix = np.zeros((2, 2), dtype=int)
outputs_all = torch.tensor([])
predicted_all = torch.tensor([]).int()
labels_all = torch.tensor([]).int()
paths_all = []

predicted_all = predicted_all.cuda()
labels_all = labels_all.cuda()
outputs_all = outputs_all.cuda()

with torch.no_grad():

    # model = net.eval()
    test_acc = 0
    time_start = time.time()

    for i, data in enumerate(test_loader, 0):
        images, labels, paths = data

        a += 1
        percentage = 100*a/itr
        # print('{}'.format(percentage))
        if percentage<100:
            print('                ',round(percentage, 1),'%')
        else:
            print('                 100.0 %')

        images = images.cuda()
        img = Variable(images)

        labels = labels.cuda()
        lbl = Variable(labels)

        outputs = net(images)

        optimizer.zero_grad()

        outputs = outputs.cuda()

        labels = labels.int()

        # from one-hot to scalar label
        _, prediction = torch.max(outputs.data, 1)
        test_acc += torch.sum(prediction == lbl.data)
        predicted = torch.max(outputs, 1)[1].int()
        predicted = predicted.cuda()

        outputs_all = torch.cat((outputs_all,outputs), 0)
        predicted_all = torch.cat((predicted_all,predicted), 0)
        labels_all = torch.cat((labels_all, labels), 0)
        paths_all += list(paths)
        optimizer.step()
    time_end = time.time()-time_start
    print('< Process Complete>')
    print('--------------------')
    print('Testing time:', round(time_end, 2), 's')
    print('Testing "Accuracy" of the selected image: {}%'.format(100 * test_acc / number_of_sub_images_in_folder))

# Confusion matrix
for i in range(labels_all.shape[0]):
    confusion_matrix[predicted_all[i], labels_all[i]] += 1
print('Confusion matrix:\n', confusion_matrix)

# confusion_matrix
mdict = {'confusion_matrix': confusion_matrix}
savename = 'result/network_4/image_86_small/test_12/confusion_matrix'
scipy.io.savemat(savename, mdict=mdict)
print('Saved to result/network_4/image_86_small/test_12/confusion_matrix.mat')

# save prediction_on_test_dataset
outputs_all = outputs_all.cpu()
predicted_all = predicted_all.cpu()
labels_all = labels_all.cpu()

outputs_all = outputs_all.numpy()
predicted_all = predicted_all.numpy()
labels_all = labels_all.numpy()
mdict = {'outputs_all': outputs_all,
        'predicted_all': predicted_all.squeeze(),
         'labels_all': labels_all.squeeze(),
         'paths_all': paths_all}
savename = 'result/network_4/image_86_small/test_12/prediction_on_test_dataset'
scipy.io.savemat(savename, mdict=mdict)
print('Saved to result/network_4/image_86_small/test_12/prediction_on_test_dataset.mat')
