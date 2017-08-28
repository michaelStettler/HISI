import numpy as np
import matplotlib.pylab as plt
import scipy.misc

# mnist = np.load('../../../Stimuli/mnist_train.npy', encoding="latin1")
mnist = np.load('../../../Stimuli/mnist_train_0bar_baseline.npy', encoding="latin1")
# mnist_test = np.load('../../../Stimuli/mnist_test.npy', encoding="latin1")
mnist_test_baseline = np.load('../../../Stimuli/mnist_test_0bar_baseline.npy', encoding="latin1")
mnist_test = np.load('../../../Stimuli/mnist_test_4bar.npy', encoding="latin1")

# mnist_fast_lami = np.load('../../../Stimuli/mnist_fast_lami_train_original.npy', encoding="latin1")
# mnist_fast_lami = np.load('../../../Stimuli/mnist_fast_lami_3bar_train_all_objects.npy', encoding="latin1")
mnist_fast_lami = np.load('../../../Stimuli/mnist_fast_lami_train_last_image.npy', encoding="latin1")
# mnist_fast_lami = np.load('../../../Stimuli/mnist_fast_lami_0bar_train.npy', encoding="latin1")

# mnist_fast_lami_test = np.load('../../../Stimuli/mnist_fast_lami_test_original.npy', encoding="latin1")
# mnist_fast_lami_test = np.load('../../../Stimuli/mnist_fast_lami_4bar_test_all_objects.npy', encoding="latin1")
mnist_fast_lami_test = np.load('../../../Stimuli/mnist_fast_lami_test_last_image.npy', encoding="latin1")
# mnist_fast_lami_test = np.load('../../../Stimuli/mnist_fast_lami_0bar_test.npy', encoding="latin1")

idx = np.load('../../../Stimuli/mnist_train_label.npy', encoding="latin1")
idx_test = np.load('../../../Stimuli/mnist_test_label.npy', encoding="latin1")

print("shape train", np.shape(mnist_fast_lami))
print("shape test", np.shape(mnist_fast_lami_test))

# test = np.load('results/mnist_fast_lami_0.npy')
# print(np.shape(test))
# for img in test:
#
#     img = np.reshape(img, (28, 28))
#     plt.figure()
#     plt.imshow(img)
#     plt.show()
# for i in [1, 44, 48, 687, 421, 4568, 1239, 46852]:
# for i in [44]:
#     print(idx[i])
#
#     img = np.reshape(mnist[i], (28, 28))
#     plt.figure()
#     plt.imshow(img)
#
#     img = np.reshape(mnist_fast_lami[i], (28, 28))
#     # img = np.reshape(mnist_fast_lami[i], (56, 28))
#     plt.figure()
#     plt.imshow(img)
#
#     plt.show()

# for i in [0, 1, 874, 4598, 1456, 3691, 548, 541]:
# for i in [22, 24, 26, 30, 62, 75, 118, 236, 380, 403]: #bad
for i in [0, 2, 7, 21, 23, 27, 59, 61, 69, 77]: #good
# for i in [541]:
    print(i, idx_test[i])

    img = np.reshape(mnist_test[i], (28, 28))
    plt.figure()
    plt.imshow(img)
    scipy.misc.toimage(1 - img, cmin=0.0, cmax=1.0).save('mnist_4bar_'+str(i)+'.jpg')

    img = np.reshape(mnist_fast_lami_test[i], (28, 28))
    # img = np.reshape(mnist_fast_lami_test[i], (56, 28))
    plt.figure()
    plt.imshow(img)
    scipy.misc.toimage(1 - img, cmin=0.0, cmax=1.0).save('mnist_fl_'+str(i)+'.jpg')

    # plt.show()

    img = np.reshape(mnist_test_baseline[i], (28, 28))
    scipy.misc.toimage(1 - img, cmin=0.0, cmax=1.0).save('mnist_0bar_'+str(i)+'.jpg')


# img = np.reshape(mnist_test[541], (28, 28))
# scipy.misc.toimage(mnist_fast_lami[44], cmin=0.0, cmax=1.0).save('fast_lami_train_all_objects.jpg')
# scipy.misc.toimage(mnist_fast_lami_test[541], cmin=0.0, cmax=1.0).save('fast_lami_test_all_objects.jpg')
# scipy.misc.toimage(mnist[44], cmin=0.0, cmax=1.0).save('0bar_train.jpg')
# scipy.misc.toimage(mnist_test[541], cmin=0.0, cmax=1.0).save('0bar_test.jpg')