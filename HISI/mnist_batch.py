import struct
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

create_batch = True

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# np.save("Stimuli/mnist_train", mnist.train.images)
# np.save("Stimuli/mnist_train_label", mnist.train.labels)
# np.save("Stimuli/mnist_test", mnist.test.images)
# np.save("Stimuli/mnist_test_label", mnist.test.labels)
#
# print("Number images in train: ", np.shape(mnist.train.images)[0])
# print("Number images in test: ", np.shape(mnist.test.images)[0])

num_pass_image = 20 # CNN1
if create_batch:
    batch = []
    #create batch for every 1000
    for i in range(1,5):
        num_images = i*1000

        temp_batch = []
        for j in range(num_pass_image):
            batch1 = np.arange(num_images)
            np.random.shuffle(batch1)
            temp_batch.append(batch1)

        temp_batch = np.reshape(temp_batch,(num_images*num_pass_image,))
        batch.append(temp_batch)

    #create batch for every 5000
    for i in range(1,12):
        num_images = i*5000

        temp_batch = []
        for j in range(num_pass_image):
            batch1 = np.arange(num_images)
            np.random.shuffle(batch1)
            temp_batch.append(batch1)

        temp_batch = np.reshape(temp_batch,(num_images*num_pass_image,))
        batch.append(temp_batch)

    np.save("Stimuli/batch", batch)

print("batch size", np.size(batch))