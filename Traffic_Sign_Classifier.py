
# coding: utf-8

# ## Project: Build a Traffic Sign Recognition Classifier

# ## ----Step 0: Load The Data----------------------
from __future__ import division
import matplotlib.pyplot as plt
import random
import numpy as np
import pickle
import csv

# TODO: Fill this in based on where you saved the training and testing data
# Load pickled data
training_file = 'data/train.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

#----READ CSV NAME FILES----
sign_names_list = []
with open('signnames.csv', 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        sign_names_list.append(line[1])

f.close


# ## ------------Step 1: Dataset Summary & Exploration------------------------
# 
# The pickled data is a dictionary with 4 key/value pairs:
#
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
#

# TODO: Number of training examples
n_train = len(y_train)

# TODO: Number of testing examples.
n_test = len(y_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_test))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# Visualize the German Traffic Signs Dataset using the pickled file(s).
#  This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections.




fig,ax = plt.subplots(2,5)
for i in range(5):
    r1 = random.randint(0,len(X_train))
    r2 = random.randint(0,len(X_test))
    img_train = X_train[r1]
    img_test = X_test[r2]
    ax[0,i].imshow(img_train)
    ax[0,i].axis("off")
    ax[0,i].set_title("X_train[%d]"%r1,fontsize=8)
    ax[1,i].imshow(img_test)
    ax[1,i].axis("off")
    ax[1,i].set_title("X_test[%d]"%r2,fontsize=8)
# plt.show()

#-----PLOT HISTOGRAM OF NUMBERS OF TRAINING IMAGE--------
# plt.figure()
# plt.hist(y_train,n_classes)
# plt.show()


# ------------------------------
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs.
# Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).
# It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.
#
# ### Implementation

#-----IMAGE PREPROCESS----------
def preprocess(images):
    for i in range(len(images)):
        img = images[i]
        for c in range(img.shape[2]):
            img[...,c] = np.float32(img[...,c]/255-0.5)
        images[i] = np.float32(img)
    return images

def one_hot_encode(y_list):
    result = np.zeros((len(y_list),n_classes),dtype="uint8")
    for i in range(len(y_list)):
        num = y_list[i]
        result[i] = [1 if num == j else 0  for j in range(n_classes)]
    return  result

X_train =np.float32(X_train)
X_test_pre= np.float32(X_test)


X_train_pre = preprocess(X_train)
y_train_pre = one_hot_encode(y_train)

X_test_pre = preprocess(X_test_pre)
y_test_pre = one_hot_encode(y_test)

X_test_pre = np.array(X_test_pre)
y_test_pre = np.array(y_test_pre)

# plt.figure()

# fig,ax = plt.subplots(2,5)
# for i in range(5):
#     r1 = random.randint(0,len(X_train_pre))
#     r2 = random.randint(0,len(X_test_pre))
#     img_train = X_train[r1]
#     img_test = X_test[r2]
#     ax[0,i].imshow(img_train)
#     ax[0,i].axis("off")
#     ax[0,i].set_title("X_train[%d]"%r1,fontsize=8)
#     ax[1,i].imshow(img_test)
#     ax[1,i].axis("off")
#     ax[1,i].set_title("X_test[%d]"%r2,fontsize=8)
# plt.show()

#-----------------------DATA SHUFFLE-------
# from sklearn.utils import shuffle
# from scipy.sparse import coo_matrix
# X_sparse = coo_matrix(np.array(X_train_pre))
# X_train_pre, X_sparse, y_train_pre = shuffle(X_train_pre, X_sparse, y_train_pre, random_state=0)
indexShuffle = np.arange(len(X_train_pre))
np.random.shuffle(indexShuffle)
indexShuffle.reshape(len(X_train_pre),1)
X_train_pre = X_train_pre[indexShuffle,...]
y_train_pre = y_train_pre[indexShuffle,...]

#-----TRAIN,VALIDATION DATASET SPLIT-------
from sklearn.model_selection import train_test_split
X_train_pre,X_valid,y_train_pre,y_valid = train_test_split(X_train_pre,y_train_pre,test_size=0.1)

print("Train size is :",len(X_train_pre))
print("Valid size is :",len(X_valid))

#-------------DEFINE TENSORFLOW GRAPH--------------
import tensorflow as tf

batch_size = 64
image_size = 32
num_channels = 3

# --------DEFINE PLACEHOLDER FOR DATASET-------
tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size,n_classes))

# # -------USE ALL VALIDATION DATA WILL RUN OUT OF MEMORY----------
# tf_valid_dataset = tf.constant(X_valid,dtype=tf.float32)
# tf_test_dataset = tf.constant(X_test_pre,dtype=tf.float32)
#tf_test_dataset = tf.constant(y_valid)

tf_valid_dataset = tf.placeholder(tf.float32,shape=(batch_size,image_size,image_size,num_channels))
# tf_valid_label = tf.placeholder()
tf_test_dataset = tf.placeholder(tf.float32,shape=(1,image_size,image_size,num_channels))

#------------CONVOLUTION PARAMETER--------
c1_depth = 32
c1_kernal_sz = 5

c2_depth = 64
c2_kernal_sz = 5

c3_depth = 120
c3_kernal_sz = 5

num_hidden = 512

# # -----------WEIGHTS DEFINE-----------
c1_weights = tf.Variable(tf.truncated_normal(
    [c1_kernal_sz, c1_kernal_sz, num_channels, c1_depth], stddev=0.1))
c1_biases = tf.Variable(tf.zeros([c1_depth]))

c2_weights = tf.Variable(tf.truncated_normal(
    [c2_kernal_sz, c2_kernal_sz, c1_depth, c2_depth], stddev=0.1))
c2_biases = tf.Variable(tf.constant(1.0, shape=[c2_depth]))

c3_weights = tf.Variable(tf.truncated_normal(
		[c3_kernal_sz, c3_kernal_sz, c2_depth, c3_depth], stddev=0.1))
c3_biases = tf.Variable(tf.constant(1.0, shape=[c3_depth]))

c3_conv_dim = (((((image_size + 1) // 2) + 1) // 2) + 1) // 2

fc_weights = tf.Variable(tf.truncated_normal(
    [c3_conv_dim * c3_conv_dim * c3_depth, num_hidden], stddev=0.1))
fc_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

out_weights = tf.Variable(tf.truncated_normal(
    [num_hidden, n_classes], stddev=0.1))
out_biases = tf.Variable(tf.constant(1.0, shape=[n_classes]))

# # ---------GRUPH MODEL--------
def conv_model(data):
    print("input data shape: ",data.get_shape().as_list())
    #---CONV 1----
    conv = tf.nn.conv2d(data, c1_weights, [1, 1, 1, 1], padding='SAME')
    pooled = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(pooled + c1_biases)
    print("conv1 shape: ",conv.get_shape().as_list())

    #--CONV 2----
    print(pooled.get_shape().as_list())
    conv = tf.nn.conv2d(hidden, c2_weights, [1, 1, 1, 1], padding='SAME')
    pooled = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(pooled + c2_biases)

    shape = pooled.get_shape().as_list()
    print("conv2 shape: ",shape)
    #----CONV 3----
    conv = tf.nn.conv2d(hidden, c3_weights, [1, 1, 1, 1], padding='SAME')
    pooled = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(pooled + c3_biases)

    shape = pooled.get_shape().as_list()
    print("conv3 shape: ",shape)
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, fc_weights) + fc_biases)
    return tf.matmul(hidden, out_weights) + out_biases

train_sign = conv_model(tf_train_dataset)
# loss or cost
loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(train_sign, tf_train_labels))

# optimizer
# optimizer = tf.train.AdagradOptimizer(0.01).minimize(loss)
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

# Predictions for the training, validation, and test data.
train_prediction = tf.nn.softmax(train_sign)
valid_prediction = tf.nn.softmax(conv_model(tf_valid_dataset))
test_prediction = tf.nn.softmax(conv_model(tf_test_dataset))


#---------HYPER PARAMETER ----EPOCHS AND STEPS-----
# num_epochs * train_size) // BATCH_SIZE,E.g.10*n_//128
num_epochs = 10
num_steps = num_epochs*len(y_train_pre)//batch_size
print("Number of training steps: ",num_steps)


#--------ACCURACY DEFINE--------------
def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
	        / predictions.shape[0])
	# return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
	#         / len(predictions))


#--------TEST ACCURACY------------



# # # ------RUN GRAPH ----------
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (len(X_train_pre) - batch_size)
        # ---train feed_dict data----
        train_batch_data = X_train_pre[offset:(offset + batch_size), :, :, :]
        train_batch_labels = y_train_pre[offset:(offset + batch_size), :]
        train_feed_dict = {tf_train_dataset: train_batch_data, tf_train_labels: train_batch_labels}

        #---valid feed_dict data----
        offset = (step * batch_size) % (len(X_valid) - batch_size)
        valid_batch_data = X_valid[offset:(offset + batch_size),...]
        valid_batch_labels = y_valid[offset:(offset + batch_size),...]
        valid_feed_dict = {tf_valid_dataset: valid_batch_data}

        # ---test feed_dict data----
        offset = (step * batch_size) % (len(X_test) - batch_size)
        test_batch_data = X_test_pre[offset:(offset + batch_size),...]
        test_batch_labels = y_test_pre[offset:(offset + batch_size),...]
        test_feed_dict = {tf_test_dataset: test_batch_data}

        # valid_test_feed_dict = {tf_valid_dataset:valid_batch_data,tf_test_dataset: test_batch_data}

        #---Train data loss and prediction----
        _, los, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=train_feed_dict)

        #-----Valid data prediction-------SHAPE PROBLEM IS FOUND WHEN DEBUG-------TRY TWO NIGHTS---F_CK----
        valid_predict = sess.run([valid_prediction],
                                    feed_dict=valid_feed_dict)
        valid_predict = np.array(valid_predict)
        valid_predict = np.reshape(valid_predict,(64,43))

        # -----Test data prediction------------
        # test_predict = sess.run([test_prediction],
        #                          feed_dict=test_feed_dict)
        # test_predict = np.array(test_predict)
        # test_predict = np.reshape(test_predict, (64, 43))

        if (step % 200 == 0):
            print('--------step %d--------' % step)
            print('Minibatch loss :%f' % los)
            print('Minibatch accuracy of TRAIN data: %.1f%%' % accuracy(predictions, train_batch_labels))

            print('Minibatch accuracy of VALIDATION data: %.1f%%' % accuracy(valid_predict,valid_batch_labels))
            # print('Minibatch accuracy of TEST data: %.1f%% ' % accuracy(test_predict, test_batch_labels))

    #         print('Validation accuracy: %.1f%%' % accuracy(
    #             valid_prediction.eval(), y_valid))
    # print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), y_test_pre))

    # Predict random selected signs
    fig,ax = plt.subplots(2,4,figsize = (16,9))
    for i in range(8):
        rand_idx = random.randint(0,len(X_test))
        sign_name = sign_names_list[y_test[rand_idx]+1]

        # predict
        test_batch_data = X_test_pre[rand_idx,...]
        test_batch_data = np.reshape(test_batch_data,(1,32,32,3))
        test_feed_dict = {tf_test_dataset: test_batch_data}
        test_predict = sess.run([test_prediction],
                                 feed_dict=test_feed_dict)
        test_predict = np.array(test_predict)
        test_predict = np.reshape(test_predict, (1, 43))
        predict_idx = np.argmax(test_predict)
        predict_name = sign_names_list[predict_idx+1]

        img_test = X_test[rand_idx]
        ax[i / 4, i % 4].imshow(img_test)
        ax[i / 4, i % 4].axis("off")
        ax[i / 4, i % 4].set_title("Ture:%s \n Predict:%s " % (sign_name, predict_name))

    plt.show()





# fig,ax = plt.subplots(2,5,figsize = (16,9))
# for i in range(10):
#     rand_idx = random.randint(0,len(X_test))
#     sign_name = sign_names_list[y_test[rand_idx]]
#
#     # predict
#     # test_batch_data = X_test_pre[rand_idx,...]
#     # test_batch_data = np.reshape(test_batch_data,(1,32,32,3))
#     # test_feed_dict = {tf_test_dataset: test_batch_data}
#     # test_predict = sess.run([test_prediction],
#     #                          feed_dict=test_feed_dict)
#     # test_predict = np.array(test_predict)
#     # test_predict = np.reshape(test_predict, (1, 43))
#     # predict_idx = np.argmax(test_predict)
#     # predict_name = sign_names_list[predict_idx]
#
#     img_test = X_test[rand_idx]
#     ax[i/5,i%5].imshow(img_test)
#     ax[i/5,i%5].axis("off")
#     ax[i/5,i%5].set_title("Ture:%s \n Predict:%s " %(sign_name,"dfdsf"))
#
# plt.show()


