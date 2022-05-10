#add your imports here
import bounding_box
import predict_function as pf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 
from sys import argv
from glob import glob
import numpy as np
import os
import operator
# variables
steps = 5000
batchSize = 124
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


sy = ['dot', 'tan', ')', '(', '+', '-', '|', 'sqrt', '1', '0', '3', '2', '4', '6','8', 'mul', 'pi', 'sin', 'A', 'cube root', 'co', 'os', 'mn', 'frac', 'cos', 'delta', 'a', 'c', 'b', 'bar', 'd', 'f', 'i', 'h', 'k', 'm', 'o', 'n', 'p', 's', 't', 'y', 'x', 'z', 'v', 'l', 'w', 'div', 'z_no_line', 'z_line']

slash_sy = ['tan', 'sqrt', 'mul', 'pi', 'sin','frac', 'cos', 'delta', 'bar', 'div','^','_']

variable = ['1', '0', '3', '2', '4', '6', 'pi', 'A', 'a', 'c', 'b', 'd', 'f', 'i', 'h', 'k', 'm', 'o', 'n', 'p', 's', 't', 'y', 'x', '(', ')','|', 'mn', 'z_no_line', 'z_line']
brules = {}
for i in range(0,len(sy)):
    brules[i] = sy[i]

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 50]))
b = tf.Variable(tf.zeros([50]))

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def max_pool_1x1(x):
    return tf.nn.max_pool(x, ksize=[1, kennelSize[0], kennelSize[1], 1], strides=[1, 1, 1, 1], padding='SAME')

# First Convolutional Layer
W_conv1 = weight_variable([layer1Patch[0], layer1Patch[1], 1, layer1Feature])
b_conv1 = bias_variable([layer1Feature])

x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional Layer
W_conv2 = weight_variable([layer2Patch[0], layer2Patch[1], layer1Feature, layer2Feature])
b_conv2 = bias_variable([layer2Feature])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# THIRD LAYER
W_conv3 = weight_variable([layer2Patch[0], layer3Patch[1], layer2Feature, layer3Feature])
b_conv3 = bias_variable([layer3Feature])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_1x1(h_conv3)

# Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * layer3Feature, hiddenLayer])  # hidden layer
b_fc1 = bias_variable([hiddenLayer])  # hidden layer

h_pool3_flat = tf.reshape(h_pool3, [-1, 7 * 7 * layer3Feature])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#readout layer
W_fc2 = weight_variable([hiddenLayer, 50])
b_fc2 = bias_variable([50])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

init_op = tf.initialize_all_variables()
saver = tf.train.Saver()

class SymPred():
    def __init__(self,prediction, x1, y1, x2, y2):
        self.prediction = prediction
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    def __str__(self):
        return self.prediction + '\t' + '\t'.join([
                                                str(self.x1),
                                                str(self.y1),
                                                str(self.x2),
                                                str(self.y2)])

def predict(image_path):
    test_symbol_list = bounding_box.createSymbol(image_path)
    test_symbol_list = sorted(test_symbol_list, key=operator.itemgetter(2, 3))
    pre_symbol_list = []
    for i in range(len(test_symbol_list)):
        test_symbol = test_symbol_list[i]
        imvalue, _ = pf.prepare_image(test_symbol[0])
        prediction = tf.argmax(y_conv, 1)
        
        predint = prediction.eval(feed_dict={x: [imvalue], keep_prob: 1.0}, session=sess)

        if test_symbol[1] != "dot":
            predict_result = brules[predint[0]]
        else:
            predict_result = "dot"
        test_symbol = (test_symbol[0], predict_result, test_symbol[2], test_symbol[3], test_symbol[4], test_symbol[5])
        test_symbol_list[i] = test_symbol
        nf.write("\t%s\t[%d, %d, %d, %d]\n" %(test_symbol[1], test_symbol[2], test_symbol[3], test_symbol[4], test_symbol[5])) 

    updated_symbol_list = pf.update(image_path, test_symbol_list)
    
    for s in updated_symbol_list:
        pre_symbol = SymPred(s[1], s[2], s[3], s[4], s[5])
        pre_symbol_list.append(pre_symbol)

    # equation = pf.toLatex(updated_symbol_list)
    _, tail = os.path.split(image_path)

    return tail +"\t" + str(pf.categorize(updated_symbol_list))+'\n'

if __name__ == '__main__':

    image_folder_path = "./result"

    image_paths = glob(image_folder_path + '/*png')
    results = []

    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, os.getcwd()+"/model.ckpt")
        nf = open("result.txt", 'w')

        for image_path in image_paths:
            nf.write("Prediction for equation %s\n" %(image_path))
            impred = predict(image_path)
            results.append(impred)

    with open('predictions.txt','w') as fout:
        for res in results:
            fout.write(str(res))