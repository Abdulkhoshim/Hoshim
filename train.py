from __future__ import print_function

import scipy
from scipy import io
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import time
import xlrd
import xlwt
from xlutils.copy import copy

import paras
from network_small_size import Net

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
mkdir('model/network_4')

# GPU
USE_GPU = paras.USE_GPU

# regularzation
USE_REG = paras.USE_REG

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

# data load
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
train_set = torchvision.datasets.ImageFolder(paras.train_image_86_small, transform=transform) #paras.train_image?
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# network
net = Net()
net = net.cuda()

# # CrossEntropy
# criterion = nn.CrossEntropyLoss()
# SGD
# optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

# train
loss_all = []
l1 = 0
l2 = 0
time_start_whole = time.time()

# # Printing Accuraccy
# loss_fn = nn.CrossEntropyLoss()

# # Adam
# optimizer = optim.Adam(net.parameters(), lr=lr)
number_of_images_in_folder = len(train_loader.dataset)
print(number_of_images_in_folder)
location = 'model/network_4/excell_files/image_86_small/test.xls'
start_time_overall = time.time()
# number_of_images_in_folder = len([f for f in os.listdir('sub_image_train/0_background') if os.path.isfile(os.path.join('sub_image_train/0_background', f))])+ len([f for f in os.listdir('sub_image_train/1_crack') if
#                                       os.path.isfile(os.path.join('sub_image_train/1_crack', f))])
# create excell file for recording data
if os.path.exists(location):
    rb = xlrd.open_workbook(location, formatting_info=True)
    r_sheet = rb.sheet_by_index(0)
    wb = copy(rb)
    sheet = wb.get_sheet(0)
    sheet.write(0, 0, "Epoch")
    sheet.write(0, 1, "Iteration")
    sheet.write(0, 2, "Loss")
    wb.save(location)
else:
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Sheet_1')
    wb.save(location)
    rb = xlrd.open_workbook(location, formatting_info=True)
    r_sheet = rb.sheet_by_index(0)
    wb = copy(rb)
    sheet = wb.get_sheet(0)
    sheet.write(0, 0, "Epoch")
    sheet.write(0, 1, "Iteration")
    sheet.write(0, 2, "Loss")
    wb.save(location)

excel_c_1 = 0
excel_c_2 = 0
excel_c_3 = 0
num = 0
x_round = 0

for e in range(epoch):
    NET = net.train()
    train_acc = 0.0
    train_loss = 0.0
    excel_c_1 += 1
    excel_c_2 += 1
    num += 1
    itr = 0
    x = number_of_images_in_folder / paras.batch_size
    time_start_epoch = time.time()

    if (e<3):
        lr = paras.lr
    elif(e>=3 and e<7):
        lr = paras.lr/10
    elif(e>=7 and e<14):
        lr = paras.lr/20
    elif(e>=14 and e<20):
        lr = paras.lr/50
    elif (e >= 20):
        lr = paras.lr /100

    # Adam
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # CrossEntropy
    criterion = nn.CrossEntropyLoss()
    # Printing Accuraccy
    loss_fn = nn.CrossEntropyLoss()

    # running_loss = 0.0
    for i, data in enumerate(train_loader, 0):

        a = 236
        itr += 1
        time_start_iter = time.time()

        inputs, labels = data  # zero the parameter gradients

        img = Variable(inputs.cuda())   #
        label = Variable(labels.cuda()) #
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        # forward
        outputs = net.forward(inputs)
        output = NET.forward(img)       #
        outputs = outputs.cuda()

        loss = loss_fn(outputs, label)
        # Backpropagate the loss
        loss.backward()
        optimizer.step()

        running_loss = loss.item()
        loss_all.append(running_loss)

        train_loss += loss.cpu().data[0] * img.size(0)
        _, prediction = torch.max(output.data, 1)

        train_acc += torch.sum(prediction == label.data)

        time_end_iter = time.time() - time_start_iter
        # print('loss:', running_loss)
        print('[Epoch %d, %5d], Loss: %.6f, Time Cost: %.4f s' % (e + 1, i + 1, running_loss, time_end_iter))
        accuracy = 100 * train_acc / number_of_images_in_folder
        # accuracy.item()
        # print(accuracy.item())
        # running_loss = 0.0
        # if i % 2 == 0:
        print('              '
              'Progress: [{}/{}] .............<{:.0f}%>'.format(
            i * len(data), len(train_loader.dataset),
            100. * i / len(train_loader), ))
        print('              Training Accuracy on each Iteration: {}%'.format(
                                                                            100 * train_acc / number_of_images_in_folder))

        # x = number_of_images_in_folder/paras.batch_size
        # x_round = round(x)
        # x_round = 0
        # print(x_round)

        sheet.write(itr + x_round, 0, num)
        sheet.write(itr + x_round, 1, running_loss)
        sheet.write(itr + x_round, 2, accuracy.item())
        wb.save(location)

    x_round = x_round + round(x) + 1

    # Compute the average acc and loss over "all training images"
    train_accur = 100 * train_acc / number_of_images_in_folder
    train_losses = train_loss / number_of_images_in_folder

    # save each epochs loss in excell
    # sheet.write(237, 2, train_losses)
    # wb.save('model/network_4/excell_files/test.xls')

    # save model
    torch.save(net.state_dict(), 'model/network_4/image_86_small/params_' + str(e+1).zfill(2) + '.pkl')
    torch.save(net, 'model/network_4/image_86_small/model_' + str(e+1).zfill(2) + '.pkl')
    print(' ')
    print('Finished training, model was saved to model/network_4/image_86_small/'
          '')
    time_end_epoch = time.time() - time_start_epoch
    print('Epoch %d time cost: %.2f s' % (e + 1, time_end_epoch))
    print("Train Accuracy: {}%, Train Loss: {}:".format(train_accur, train_losses))
    print(' ')

# save loss_all
mdict = {'loss_all': loss_all}
savename = 'result/network_4/image_86_small/loss_all'
scipy.io.savemat(savename, mdict=mdict)
time_end_overall = time.time() - start_time_overall
print('Overal Training Time Cost: {} s.'.format(time_end_overall))
print('Saved to result/network_4/image_86_small/loss_all.mat')

plt.figure()
plt.plot(np.arange(0, len(loss_all)), loss_all)
plt.show()

