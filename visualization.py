from __future__ import print_function

import scipy
from scipy import io
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import paras

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

outputs_all = scipy.io.loadmat('result/network_4/image_86_small/test_12/prediction_on_test_dataset',mdict=None, appendmat=True)['outputs_all']
predicted_all = scipy.io.loadmat('result/network_4/image_86_small/test_12/prediction_on_test_dataset', mdict=None, appendmat=True)['predicted_all'].squeeze()
labels_all = scipy.io.loadmat('result/network_4/image_86_small/test_12/prediction_on_test_dataset', mdict=None, appendmat=True)['labels_all'].squeeze()
paths_all = scipy.io.loadmat('result/network_4/image_86_small/test_12/prediction_on_test_dataset', mdict=None, appendmat=True)['paths_all']

# to probability
# outputs_all = F.softmax(torch.from_numpy(outputs_all), dim=1)

# visualization
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
#test_set = ImageFolderWithPaths(paras.test_image_01, transform=transform)
test_set = ImageFolderWithPaths(paras.test_image_12, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

# for overlap visualization(probability map)
result_2images = np.zeros((h, w, 2))
for i in range(predicted_all.size):
    prediction = predicted_all[i]
    path = paths_all[i]
    sub_image_row = int(path.split('\\')[-1].split('.')[0].split('_')[2])
    sub_image_column = int(path.split('\\')[-1].split('.')[0].split('_')[3])
    range_row_lower = sub_image_row * stride
    range_column_lower = sub_image_column * stride
    range_row_upper = range_row_lower + sub_image_size
    range_column_upper = range_column_lower + sub_image_size
    result_2images[range_row_lower:range_row_upper, range_column_lower:range_column_upper, prediction] += 1
    result_2images[range_row_lower:range_row_upper, range_column_lower:range_column_upper, :] += outputs_all[i, :]

result_2images = result_2images / result_2images.max()
result_2images *= 255
result_2images = result_2images.astype(int)

#crack
crack = result_2images[:, :, 1]
crack_colored = np.zeros((h, w, 3))
crack_colored[:, :, 1] = crack
crack_colored = np.array(crack_colored, dtype=int)

#background
bg = result_2images[:, :, 0]
bg_colored = np.zeros((h, w, 3))
bg_colored[:, :, 0] = bg
bg_colored = np.array(bg_colored, dtype=int)

cv2.imwrite('result/network_4/image_86_small/test_12/'+ 'bg.jpg', bg)
cv2.imwrite('result/network_4/image_86_small/test_12/'+ 'cr.jpg', crack)

plt.figure()
# plt.subplot(1,2,1)
# plt.title('crack')
# plt.imshow(crack)

plt.subplot(1,1,1)
plt.title('CRACK (red color)')
plt.imshow(crack_colored*255)
# plt.imshow(crack_colored,vmin=0, vmax=255)plt.imshow(crack*255)

# plt.subplot(1,2,2)
# plt.title('BACKGROUND (red color)')
# # plt.imshow((bg*255).astype(int))
# plt.imshow(bg_colored*255)

plt.savefig('result/network_4/image_86_small/test_12/' + 'result.jpg')
plt.show()