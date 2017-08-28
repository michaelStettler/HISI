import numpy as np
import matplotlib.pyplot as plt

train_data = np.load("MNIST/mnist_train.npy")
test_data = np.load("MNIST/mnist_test.npy")


train = []
train_bar = []
test = []
test_bar = []
num_bar = 3
size_bar = int(28 / (2 * num_bar + 1))
color_number = .4
color_bar = 1.

for j in range(np.shape(train_data)[0]):
    img = np.copy(train_data[j])
    img = img.reshape(28, 28)

    img[img > 0] = color_number
    train.append(img)

    img2 = np.copy(img)

    for k in range(int(2 * num_bar + 1)):
        if k % 2 == 1:
            start = k * size_bar
            img2[start:start + size_bar, :] = color_bar

    train_bar.append(img2)

for j in range(np.shape(test_data)[0]):
    img = np.copy(test_data[j])
    img = img.reshape(28, 28)

    img[img > 0] = color_number
    test.append(img)

    img2 = np.copy(img)

    for k in range(int(2 * num_bar + 1)):
        if k % 2 == 0:
            start = k * size_bar
            img2[start:start + size_bar, :] = color_bar

    test_bar.append(img2)


np.save("Stimuli/mnist_train", train)
np.save("Stimuli/mnist_train_"+ str(int(num_bar)) +"bar", train_bar)
np.save("Stimuli/mnist_test", test)
np.save("Stimuli/mnist_test_"+ str(int(num_bar + 1)) +"bar", test_bar)

