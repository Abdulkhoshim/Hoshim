# encoding：utf-8
import numpy as np
import cv2
import scipy
from scipy import io
import glob
import os

import paras


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)



def generate_train_data(raw_image_idx, pix_label_idx):

    classes = ['0_background', '1_crack']

    raw_image = cv2.imread(raw_image_idx)
    pix_label = cv2.imread(pix_label_idx)

    h, w, c = raw_image.shape
    sub_image_size = paras.sub_image_size

    # class 1
    white = [[225, 225, 225], [255, 255, 255]]

    lower_white = np.array(white[0], dtype="uint8")
    upper_white = np.array(white[1], dtype="uint8")
    mask1 = cv2.inRange(pix_label, lower_white, upper_white)


    # for background
    stride = paras.stride_bg #cropping distance between two sub-images for non-crack images
    rows = int((h-sub_image_size)/stride + 1)
    columns = int((w-sub_image_size)/stride + 1)

    for column in np.arange(0, columns, 1):
        for row in np.arange(0, rows, 1):
            sub_image = raw_image[row*stride:row*stride+sub_image_size,\
                        column*stride:column*stride+sub_image_size, :]

            sub_image_pix_label = mask1[row * stride:row * stride + sub_image_size,
                                   column * stride:column * stride + sub_image_size]

            if sub_image_pix_label.sum()>100*255:
                sub_image_class_label = 1
            else:
                sub_image_class_label = 0

            if sub_image_class_label == 0:
                basename = os.path.basename(raw_image_idx).split('.')[0]
                savename = paras.train_image_86_small + '/' + classes[sub_image_class_label] + '/' + basename + '_' + str(row).zfill(2) + '_' \
                           + str(column).zfill(2) + '.jpg'
                            #"paras.train_image_vgg16" place (a folder)
                            # where the cropped images will be placed

                # save subimage
                cv2.imwrite(savename, sub_image)


    # for crack
    stride = paras.stride_crack
    rows = int((h-sub_image_size)/stride + 1)
    columns = int((w-sub_image_size)/stride + 1)

    for column in np.arange(0, columns, 1):
        for row in np.arange(0, rows, 1):
            sub_image = raw_image[row*stride:row*stride+sub_image_size,\
                        column*stride:column*stride+sub_image_size, :]

            sub_image_pix_label = mask1[row * stride:row * stride + sub_image_size,
                                   column * stride:column * stride + sub_image_size]

            if sub_image_pix_label.sum()>100*255:
                sub_image_class_label = 1
            else:
                sub_image_class_label = 0

            if sub_image_class_label == 1:
                basename = os.path.basename(raw_image_idx).split('.')[0]
                savename = paras.train_image_86_small + '/' + classes[sub_image_class_label]\
                           + '/' + basename + '_' + str(row).zfill(2) + '_' \
                           + str(column).zfill(2) + '.jpg'

                # save subimage
                cv2.imwrite(savename, sub_image)

# #for labeled crack
# def generate_test_data(raw_image_idx, pix_label_idx):
#
#     classes = ['0_background', '1_crack']
#
#     raw_image = cv2.imread(raw_image_idx)
#     pix_label = cv2.imread(pix_label_idx)
#
#     h, w, c = raw_image.shape
#     sub_image_size = paras.sub_image_size
#     stride = paras.stride
#     rows = int((h-sub_image_size)/stride + 1)
#     columns = int((w-sub_image_size)/stride + 1)
#
#     # class 1
#     white = [[225, 225, 225], [255, 255, 255]]
#
#     lower_white = np.array(white[0], dtype="uint8")
#     upper_white = np.array(white[1], dtype="uint8")
#     mask1 = cv2.inRange(pix_label, lower_white, upper_white)
#
#
#     for column in np.arange(0, columns, 1):
#         for row in np.arange(0, rows, 1):
#             sub_image = raw_image[row*stride:row*stride+sub_image_size, column*stride:column*stride+sub_image_size, :]
#
#             sub_image_pix_label = mask1[row * stride:row * stride + sub_image_size,
#                                    column * stride:column * stride + sub_image_size]
#
#             if sub_image_pix_label.sum()>100*255:
#                 sub_image_class_label = 1
#             else:
#                 sub_image_class_label = 0
#
#             basename = os.path.basename(raw_image_idx).split('.')[0]
#             savename = paras.test_image_04 + '/' + classes[sub_image_class_label]\
#                        + '/' + basename + '_' + str(row).zfill(2) + '_' \
#                        + str(column).zfill(2) + '.jpg'
#
#             # save subimage
#             cv2.imwrite(savename, sub_image)
#
#
# #for dfrnt image "test_01"
# def generate_test_data_predic_test_01(raw_image_idx):
#
#     raw_image = cv2.imread(raw_image_idx)
#
#     mkdir('sub_test_image/new_prediction_32/test_01/class0')
#
#     h, w, c = raw_image.shape
#     sub_image_size = paras.sub_image_size
#     stride = paras.stride
#     rows = int((h-sub_image_size)/stride + 1)
#     columns = int((w-sub_image_size)/stride + 1)
#
#     for column in np.arange(0, columns, 1):
#         for row in np.arange(0, rows, 1):
#             sub_image = raw_image[row*stride:row*stride+sub_image_size, column*stride:column*stride+sub_image_size, :]
#
#             savename = 'sub_test_image/new_prediction_32/test_01/class0/test_01_%s_%s.png' % (str(row).zfill(3), str(column).zfill(3))
#
#             # save subimage
#             cv2.imwrite(savename, sub_image)
#
#
# #for dfrnt image "test_02"
# def generate_test_data_predic_test_02(raw_image_idx):
#
#     raw_image = cv2.imread(raw_image_idx)
#
#     mkdir('sub_test_image/new_prediction_32/test_02/class0')
#
#     h, w, c = raw_image.shape
#     sub_image_size = paras.sub_image_size
#     stride = paras.stride
#     rows = int((h-sub_image_size)/stride + 1)
#     columns = int((w-sub_image_size)/stride + 1)
#
#     for column in np.arange(0, columns, 1):
#         for row in np.arange(0, rows, 1):
#             sub_image = raw_image[row*stride:row*stride+sub_image_size, column*stride:column*stride+sub_image_size, :]
#
#             savename = 'sub_test_image/new_prediction_32/test_02/class0/test_02_%s_%s.png' % (str(row).zfill(3), str(column).zfill(3))
#
#             # save subimage
#             cv2.imwrite(savename, sub_image)
#
#
#
# #for dfrnt image "test_03"
# def generate_test_data_predic_test_03(raw_image_idx):
#
#     raw_image = cv2.imread(raw_image_idx)
#
#     mkdir('sub_test_image/new_prediction_32/test_03/class0')
#
#     h, w, c = raw_image.shape
#     sub_image_size = paras.sub_image_size
#     stride = paras.stride
#     rows = int((h-sub_image_size)/stride + 1)
#     columns = int((w-sub_image_size)/stride + 1)
#
#     for column in np.arange(0, columns, 1):
#         for row in np.arange(0, rows, 1):
#             sub_image = raw_image[row*stride:row*stride+sub_image_size, column*stride:column*stride+sub_image_size, :]
#
#             savename = 'sub_test_image/new_prediction_32/test_03/class0/test_02_%s_%s.png' % (str(row).zfill(3), str(column).zfill(3))
#
#             # save subimage
#             cv2.imwrite(savename, sub_image)
#
#
# #for dfrnt image "test_04"
# def generate_test_data_predic_test_04(raw_image_idx):
#
#     raw_image = cv2.imread(raw_image_idx)
#
#     mkdir('sub_test_image/new_prediction_32/test_04/class0')
#
#     h, w, c = raw_image.shape
#     sub_image_size = paras.sub_image_size
#     stride = paras.stride
#     rows = int((h-sub_image_size)/stride + 1)
#     columns = int((w-sub_image_size)/stride + 1)
#
#     for column in np.arange(0, columns, 1):
#         for row in np.arange(0, rows, 1):
#             sub_image = raw_image[row*stride:row*stride+sub_image_size, column*stride:column*stride+sub_image_size, :]
#
#             savename = 'sub_test_image/new_prediction_32/test_04/class0/test_02_%s_%s.png' % (str(row).zfill(3), str(column).zfill(3))
#
#             # save subimage
#             cv2.imwrite(savename, sub_image)


# # for dfrnt image "test_05"
# def generate_test_data_predic_test_05(raw_image_idx):
#     raw_image = cv2.imread(raw_image_idx)
#
#     mkdir('test_image/test_05/class0')
#
#     h, w, c = raw_image.shape
#     sub_image_size = paras.sub_image_size
#     stride = paras.stride
#     rows = int((h - sub_image_size) / stride + 1)
#     columns = int((w - sub_image_size) / stride + 1)
#
#     for column in np.arange(0, columns, 1):
#         for row in np.arange(0, rows, 1):
#             sub_image = raw_image[row * stride:row * stride + sub_image_size,
#                         column * stride:column * stride + sub_image_size, :]
#
#             savename = 'test_image/test_05/class0/test_02_%s_%s.png' % (str(row).zfill(3), str(column).zfill(3))
#
#             # save subimage
#             cv2.imwrite(savename, sub_image)
#
#
# # for dfrnt image "test_06"
# def generate_test_data_predic_test_06(raw_image_idx):
#     raw_image = cv2.imread(raw_image_idx)
#
#     mkdir('test_image/test_06/class0')
#
#     h, w, c = raw_image.shape
#     sub_image_size = paras.sub_image_size
#     stride = paras.stride
#     rows = int((h - sub_image_size) / stride + 1)
#     columns = int((w - sub_image_size) / stride + 1)
#
#     for column in np.arange(0, columns, 1):
#         for row in np.arange(0, rows, 1):
#             sub_image = raw_image[row * stride:row * stride + sub_image_size,
#                         column * stride:column * stride + sub_image_size, :]
#
#             savename = 'test_image/test_06/class0/test_02_%s_%s.png' % (str(row).zfill(3), str(column).zfill(3))
#
#             # save subimage
#             cv2.imwrite(savename, sub_image)


if __name__ == '__main__':

    raw_image_list = glob.glob('image/*.jpg')
    pix_label_list = glob.glob('label/*.png')
    raw_image_list.sort()
    pix_label_list.sort()

    classes = ['0_background', '1_crack']
    sub_image_size = paras.sub_image_size

    # # generate test data for prediction image "test_01"
    # test_image_fn = 'dfrtn_img/raw_image/test_01.jpg'
    # generate_test_data_predic_test_01(test_image_fn)
    #
    # # generate test data for prediction image "test_02"
    # test_image_fn = 'dfrtn_img/raw_image/test_02.jpg'
    # generate_test_data_predic_test_02(test_image_fn)
    #
    # # generate test data for prediction image "test_03"
    # test_image_fn = 'dfrtn_img/raw_image/test_03.jpg'
    # generate_test_data_predic_test_03(test_image_fn)
    #
    # # generate test data for prediction image "test_04"
    # test_image_fn = 'dfrtn_img/raw_image/test_04.jpg'
    # generate_test_data_predic_test_04(test_image_fn)

    # # generate test data for prediction image "test_05"
    # test_image_fn = 'image_to_test/test_05.JPG'
    # generate_test_data_predic_test_05(test_image_fn)
    #
    # # generate test data for prediction image "test_06"
    # test_image_fn = 'image_to_test/test_06.JPG'
    # generate_test_data_predic_test_06(test_image_fn)

    # # generate train data
    # for i in classes:
    #     file_name = paras.train_image + '/' + str(i)
    #     # mkdir(file_name)
    #     mkdir(file_name)
    # for k in np.arange(len(raw_image_list)):
    #     raw_image_idx = raw_image_list[k]
    #     pix_label_idx = pix_label_list[k]
    #     generate_train_data(raw_image_idx, pix_label_idx)

    # generate train data
    for i in classes:
        file_name = paras.train_image_86_small + '/' + str(i)
        mkdir(file_name)
    for k in np.arange(len(raw_image_list)):
        raw_image_idx = raw_image_list[k]
        pix_label_idx = pix_label_list[k]
        generate_train_data(raw_image_idx, pix_label_idx)


    # # generate test data for lebeled image 'test_04'
    # for i in classes:
    #     file_name = paras.test_image + '/' + str(i)
    #     mkdir(file_name)
    # # change 'test_image' in 'paras.py' if you want to test other image
    # idx = paras.test_image.split('/')[1]
    # raw_image_idx = 'image/' + idx + '.jpg'
    # pix_label_idx = 'label/' + idx + '.png'
    # generate_test_data(raw_image_idx, pix_label_idx)



    # mkdir('model/crack_43')
    # mkdir('result/crack_43')
    #
    # mkdir('model/test_01')
    # mkdir('result/test_01')
    #
    # mkdir('model/test_02')
    # mkdir('result/test_02')
    #
    # mkdir('model/test_03')
    # mkdir('result/test_03')
