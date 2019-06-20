# GPU
USE_GPU = True

# regularzation
USE_REG = False


# image parameters
h, w, c = 2304, 3456, 3
sub_image_size = 32
stride = 16
stride_bg = 112
stride_crack = 42
rows = int((h - sub_image_size) / stride + 1)
columns = int((w - sub_image_size) / stride + 1)

# image folder
train_image = 'sub_image_train_74/'
train_image_18 = 'sub_image_train_18/'
train_image_23 = 'sub_image_train_23/'
train_image_30 = 'sub_image_train_30/'
train_image_44 = 'sub_image_train_44/'
train_image_57 = 'sub_image_train_57/'
train_image_80 = 'sub_image_train_80/'
train_image_80_small = 'sub_image_train_80_small/'
train_image_86_small = 'sub_image_train_86_small/'
train_image_vgg16 = 'sub_image_train_vgg16/'

'''
change this if you want to test other image, do not forget to generate test data again using 'pre_processing.py', 
once you have generated training data, you can add annotations before each line related to generating training data.
'''
test_image = 'sub_test_image/new/test_01'

test_image_01 = 'test_image/with_label/crack_43'
test_image_02 = 'test_image/with_label/crack_63'
test_image_03 = 'test_image/with_label/crack_64'
test_image_04 = 'sub_test_image/test_04'
# test_image_prd_test_01 = 'test_image/test_01'
# test_image_prd_test_02 = 'test_image/test_02'
# test_image_prd_test_03 = 'test_image/test_03'
# test_image_prd_test_04 = 'test_image/test_04'
# test_image_prd_test_05 = 'test_image/test_05'
# test_image_prd_test_06 = 'test_image/test_06'

#new predictions
test_image_prd_test_01 = 'sub_test_image/new_prediction/test_01'
test_image_prd_test_02 = 'sub_test_image/new_prediction/test_02'
test_image_prd_test_03 = 'sub_test_image/new_prediction/test_03'
test_image_prd_test_04 = 'sub_test_image/new_prediction/test_04'

test_image_07 = 'dfrtn_img/sub_test_image/test_07'
test_image_08 = 'dfrtn_img/sub_test_image/test_08'
test_image_09 = 'dfrtn_img/sub_test_image/test_09'
test_image_10 = 'dfrtn_img/sub_test_image/test_10'
test_image_11 = 'dfrtn_img/sub_test_image/test_11'
test_image_12 = 'dfrtn_img/sub_test_image/test_12'



# hyper-parameters
lr = 1e-4
momentum=0.9

epoch = 20


batch_size = 320


lambda_l1 = 0.0001
lambda_l2 = 0.00002