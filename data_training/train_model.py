import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

import pickle
import random

steps = 5000
batchSize = 100
convolution = (1, 1)
kennelSize = (2, 2)
maxPoll = (2, 2)

learningRate = 0.001
layer1Feature = 16
layer1Patch = 5, 5
layer2Feature = 32
layer2Patch = 5, 5
hiddenLayer = 100 

dropoffRate = 0.5 
layer3Feature = 64
layer3Patch = 5, 5

pf = open("./train_data.pkl","rb")
data = pickle.load(pf)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 50])
W = tf.Variable(tf.zeros([784, 50]))
b = tf.Variable(tf.zeros([50]))
y = tf.matmul(x, W) + b

def weight_variable(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
  return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def max_pool_1x1(x):
    return tf.nn.max_pool(x, ksize=[1, kennelSize[0], kennelSize[1], 1], strides=[1, 1, 1, 1], padding='SAME')

#First convolutional layer
W_conv1 = weight_variable([layer1Patch[0], layer1Patch[1], 1, layer1Feature])
b_conv1 = bias_variable([layer1Feature])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Second convolutional layer
W_conv2 = weight_variable([layer2Patch[0], layer2Patch[1], layer1Feature, layer2Feature])
b_conv2 = bias_variable([layer2Feature])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# THIRD LAYER
W_conv3 = weight_variable([layer2Patch[0], layer3Patch[1], layer2Feature, layer3Feature])
b_conv3 = bias_variable([layer3Feature])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_1x1(h_conv3)

#densely connected layer
W_fc1 = weight_variable([7 * 7 * layer3Feature, hiddenLayer])  # hidden layer
b_fc1 = bias_variable([hiddenLayer])  # hidden layer

h_pool3_flat = tf.reshape(h_pool3, [-1, 7 * 7 * layer3Feature])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#readout layer
W_fc2 = weight_variable([hiddenLayer, 50])
b_fc2 = bias_variable([50])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Define loss and optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())

# train and evaluate
for i in range(steps):
  
  batch = random.sample(list(data["images"]), 50)
  batch_images = []
  batch_labels = []

  for item in batch:
      batch_images.append(data["images"][item])
      batch_labels.append(data["labels"][item])

  if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x: batch_images, y_: batch_labels, keep_prob: 1.0})
      print("step %d, training accuracy %g"%(i, train_accuracy))

  train_step.run(feed_dict={x: batch_images, y_: batch_labels, keep_prob: dropoffRate})

save_path = saver.save(sess, "model.ckpt")
print ("Model saved in file: ", save_path)

batch_images = []
batch_labels = []
for item in data["images"]:
    batch_images.append(data["images"][item])
    batch_labels.append(data["labels"][item])

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: batch_images , y_: batch_labels, keep_prob: 1.0}))
