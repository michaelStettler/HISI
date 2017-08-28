import sys
import numpy as np
import tensorflow as tf
import datetime

#Start computational time counter
startSimulationTime = datetime.datetime.now()

#Conditions
# 0 do with the full stack
# 1 do the loop in function of the number of images
# 2 do the loop in function of the threshold
condition = 1

#Load data
#Use with the numpy array
train_cond = "mnist_train_3bar"
test_cond = "mnist_test_4bar"

if len(sys.argv) > 2:
    train_cond = sys.argv[1]
    test_cond = sys.argv[2]

mnist = np.load("Stimuli/"+train_cond+".npy", encoding="latin1")
mnist_test = np.load("Stimuli/"+test_cond+".npy", encoding="latin1")

mnist = mnist.reshape((np.shape(mnist)[0],784))
mnist_test = mnist_test.reshape((np.shape(mnist_test)[0],784))

idx = np.load('MNIST/mnist_train_label.npy', encoding="latin1")
idx_test = np.load('MNIST/mnist_test_label.npy', encoding="latin1")
all_batch = np.load('Stimuli/batch.npy', encoding="latin1")
print('Data loaded Train   Image: ', np.shape(mnist), ' idx: ',np.shape(idx))
print('Data loaded Test    Image: ', np.shape(mnist_test), ' idx: ',np.shape(idx_test))
print('Data loaded batch ', np.shape(all_batch))

#Use with tensorflow tutorial
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#Functions to initialize ReLU neurons
def weight_variable(shape,n):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=n)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#Conv and pooling definitions
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def CNN(batch, mnist, idx, mnist_test, idx_test, best):
    # Start tensorflow session
    with tf.Graph().as_default(), tf.Session() as sess:

        #Define placeholder, input and output of the program
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])

        x_image = tf.reshape(x, [-1,28,28,1])

        #-------- Model Definition --------#

        ####  First convolutional Layer ####
        W_conv1 = weight_variable([5, 5, 1, 32],"w1")
        b_conv1 = bias_variable([32])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        ####  Second convolutional Layer ####
        W_conv2 = weight_variable([5, 5, 32, 64],"w2")
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        #### Fully connected layer ####
        W_fc1 = weight_variable([7 * 7 * 64, 1024],"w_fc1")
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        #### Dropout ####
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        #### Readout layer ####
        W_fc2 = weight_variable([1024, 10],"w_fc2")
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        #-------- Train and Evaluate --------#
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # sess.run(tf.global_variables_initializer()) #need to update tensorflow ?!?!?2)


        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()


        size_batch = 50
        #Used with numpy
        for i in range(int(np.shape(batch)[0]/size_batch)):
            start = i*size_batch
            end = start + size_batch

            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:mnist[batch[start:end]], y_: idx[batch[start:end]], keep_prob: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(feed_dict={x: mnist[batch[start:end]], y_: idx[batch[start:end]], keep_prob: 0.5})

        final_accuracy = []
        for i in range(int(np.shape(mnist_test)[0]/size_batch)):
            start = i*size_batch
            end = start + size_batch

            final_accuracy.append(accuracy.eval(feed_dict={x: mnist_test[start:end], y_: idx_test[start:end], keep_prob: 1.0}))

        final_accuracy = np.mean(final_accuracy)
        print("test accuracy %g"%final_accuracy)

        # if final_accuracy > best:
        #     saver.save(sess, 'ModelWeights/CNN2_noise_0.9dp_55k')
        #     print("model saved")

        sess.close()
        del sess

        return final_accuracy

total_accuracy = []

if condition == 0:
    accuracy = CNN(all_batch[-1], mnist, idx, mnist_test, idx_test)
    #accuracy = CNN(all_batch[0], mnist, idx, mnist_test, idx_test)
    total_accuracy.append(accuracy)

elif condition == 1:
    for batch in all_batch:
        maxi = 0
        first = True
        for j in range(15):
            accuracy = CNN(batch, mnist, idx, mnist_test, idx_test, maxi)
            print("accuracy :", accuracy)
            if accuracy > maxi:
                if first:
                    total_accuracy.append(accuracy)
                    maxi = accuracy
                    first = False
                else:
                    total_accuracy[-1] = accuracy
                    maxi = accuracy

# Old paradigm
# elif condition == 2:
#     for i in range(10):
#         maxi = 0
#         first = True
#
#         thresh = float(i)/10
#         mnist = np.load('../../../Stimuli/mnist_train_thresh_'+str(thresh)+'.npy')
#         mnist_test = np.load('../../../Stimuli/mnist_test_thresh_'+str(thresh)+'.npy')
#
#         mnist = mnist.reshape((np.shape(mnist)[0], 784))
#         mnist_test = mnist_test.reshape((np.shape(mnist_test)[0], 784))
#
#         for j in range(3):
#             accuracy = CNN(all_batch[-1], mnist, idx, mnist_test, idx_test, maxi)
#             print("accuracy :", accuracy)
#
#             if accuracy > maxi:
#                 if first:
#                     total_accuracy.append(accuracy)
#                     maxi = accuracy
#                     first = False
#                 else:
#                     total_accuracy[-1] = accuracy
#                     maxi = accuracy
else:
    print("Please enter a valid condition")

print(total_accuracy)
np.save("results/accuracy_"+train_cond+"_"+test_cond, total_accuracy)


#End computational time counter
endSimulationTime = datetime.datetime.now()
print("Simulation time: " + str(endSimulationTime - startSimulationTime))

