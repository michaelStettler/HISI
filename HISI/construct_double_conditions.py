import numpy as np

train = np.load("Stimuli/mnist_train.npy")
train_3bar = np.load("Stimuli/mnist_train_3bar.npy")
train_3bar_HISI = np.load("Stimuli/mnist_train_3bar_HISI.npy")


test = np.load("Stimuli/mnist_test.npy")
test_4bar = np.load("Stimuli/mnist_test_4bar.npy")
test_4bar_HISI = np.load("Stimuli/mnist_test_4bar_HISI.npy")


mnist_double_train_train = []
mnist_double_train_3bar_train = []
mnist_double_train_3bar_train_3bar = []
mnist_double_train_3bar_HISI = []
for i, img in enumerate(train):
    #train train
    img = np.reshape(img, (28, 28))
    output = [img, img]
    output = np.reshape(output, (np.shape(output)[0] * np.shape(output)[1], np.shape(output)[2]))
    output[output < 0] = 0
    mnist_double_train_train.append(output)

    # train 3bar train
    output2 = [train_3bar[i], img]
    output2 = np.reshape(output2, (np.shape(output2)[0] * np.shape(output2)[1], np.shape(output2)[2]))
    output2[output2 < 0] = 0
    mnist_double_train_3bar_train.append(output2)

    # train 3bar train 3bar
    output3 = [train_3bar[i], train_3bar[i]]
    output3 = np.reshape(output3, (np.shape(output3)[0] * np.shape(output3)[1], np.shape(output3)[2]))
    output3[output3 < 0] = 0
    mnist_double_train_3bar_train_3bar.append(output3)

    # train 3bar HISI
    output4 = [train_3bar[i], train_3bar_HISI[i]]
    output4 = np.reshape(output4, (np.shape(output4)[0] * np.shape(output4)[1], np.shape(output4)[2]))
    output4[output4 < 0] = 0
    mnist_double_train_3bar_HISI.append(output4)

np.save("Stimuli/mnist_double_train_train", mnist_double_train_train)
np.save("Stimuli/mnist_double_train_3bar_train", mnist_double_train_3bar_train)
np.save("Stimuli/mnist_double_train_3bar_train_3bar", mnist_double_train_3bar_train_3bar)
np.save("Stimuli/mnist_double_train_3bar_HISI", mnist_double_train_3bar_HISI)

mnist_double_test_test = []
mnist_double_test_4bar_test = []
mnist_double_test_4bar_test_4bar = []
mnist_double_test_4bar_HISI = []
for i, img in enumerate(test):
    #test test
    img = np.reshape(img, (28, 28))
    output = [img, img]
    output = np.reshape(output, (np.shape(output)[0] * np.shape(output)[1], np.shape(output)[2]))
    output[output < 0] = 0
    mnist_double_test_test.append(output)

    # test 4bar test
    output2 = [test_4bar[i], img]
    output2 = np.reshape(output2, (np.shape(output2)[0] * np.shape(output2)[1], np.shape(output2)[2]))
    output2[output2 < 0] = 0
    mnist_double_test_4bar_test.append(output2)

    # test 4bar test 4bar
    output3 = [test_4bar[i], test_4bar[i]]
    output3 = np.reshape(output3, (np.shape(output3)[0] * np.shape(output3)[1], np.shape(output3)[2]))
    output3[output3 < 0] = 0
    mnist_double_test_4bar_test_4bar.append(output3)

    # test 4bar HISI
    output4 = [test_4bar[i], test_4bar_HISI[i]]
    output4 = np.reshape(output4, (np.shape(output4)[0] * np.shape(output4)[1], np.shape(output4)[2]))
    output4[output4 < 0] = 0
    mnist_double_test_4bar_HISI.append(output4)

np.save("Stimuli/mnist_double_test_test", mnist_double_test_test)
np.save("Stimuli/mnist_double_test_4bar_test", mnist_double_test_4bar_test)
np.save("Stimuli/mnist_double_test_4bar_test_4bar", mnist_double_test_4bar_test_4bar)
np.save("Stimuli/mnist_double_test_4bar_HISI", mnist_double_test_4bar_HISI)
