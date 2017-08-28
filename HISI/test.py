import numpy as np
import matplotlib.pylab as plt
from show_matrix import *
import scipy.misc


#reconstruct 2 seg mnist

# data0 = np.load("mnist4_seg0.npy")
# data1 = np.load("mnist4_seg1.npy")
#
# data = data0 + data1
#
# boundaries = np.zeros((4, np.shape(data)[0], np.shape(data)[1]))
# plt.figure()
# plt.imshow(show_matrix(data, boundaries))
#
# scipy.misc.imsave('img.jpg', 1-data)


#save "original test" mnist
# data = np.load("../../../Stimuli/mnist_test.npy")
# data = np.load("mnist_fast_lami_4bar_train.npy")
# data = np.load("results/mnist_fast_lami2_all_objects_0.npy")
# data2 = np.load("results/mnist_fast_lami2_last_objects_0.npy")
data = np.load("Stimuli/mnist_train.npy")
data2 = np.load("Stimuli/mnist_test.npy")
# labels = np.load("../../../Stimuli/mnist_test_label.npy")
# boundaries = np.zeros((4, np.shape(data)[0], np.shape(data)[1]))

for i, img in enumerate(data[0:10]):
    plt.figure()
    img[img < 0] = 0
    plt.imshow(np.reshape(img, (28,28)))

    plt.figure()
    data2[data2 < 0] = 0
    plt.imshow(np.reshape(data2[i], (28, 28)))
    plt.show()

# for i, img in enumerate(data[0:10]):
#     scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save('results/test'+str(i)+'.jpg')

# for i, img in enumerate(data[0:10]):
#     input = np.reshape(img, (56, 28))
#     input[input > 0] = 0.5
#
#     plt.figure()
#     plt.imshow(show_matrix(input, boundaries))

    # scipy.misc.imsave('mnist'+str(i)+'.jpg', 1 - input)
#
# print(np.shape(labels))
# for label in labels[0:10]:
#     print(label)
#
# plt.show()
