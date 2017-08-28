import sys
import numpy as np
import matplotlib.pylab as plt
import scipy.misc
from HISI import *
import datetime
from multiprocessing import Process

# mnist
# data = np.load("Stimuli/mnist_test.npy")
data = np.load("Stimuli/mnist_test_4bar.npy")
# data = np.load("Stimuli/mnist_train_3bar.npy")
# data = np.load("Stimuli/mnist_test_0bar_baseline.npy")
# data = np.load("Stimuli/mnist_train_0bar_baseline.npy")

show = True
obstructed = True
test = True
save = True
single_thread = False

if len(sys.argv) > 1:
    data = np.load("Stimuli/"+str(sys.argv[1])+".npy")

    #argument order: show, obstructed, test, save, single_thread
    # show
    if sys.argv[2] == 'True':
        show = True
    else:
        show = False
    # obstructed
    if sys.argv[3] == 'True':
        obstructed = True
    else:
        obstructed = False
    # test
    if sys.argv[4] == 'True':
        test = True
    else:
        test = False
    # save
    if sys.argv[5] == 'True':
        save = True
    else:
        save = False
    # single_thread
    if sys.argv[6] == 'True':
        single_thread = True
    else:
        single_thread = False

# input = np.reshape(data[0], (28, 28))     # test: 7                    train: 7
# input = np.reshape(data[1], (28, 28))     # test: 2                    train: 3
# input = np.reshape(data[2], (28, 28))     # test: 1                    train: 4
# input = np.reshape(data[3], (28, 28))     # test: 0                    train: 6
# input = np.reshape(data[4], (28, 28))     # test: 4 -> two parts       train: 1
# input = np.reshape(data[5], (28, 28))     # test: 1                    train: 8
# input = np.reshape(data[6], (28, 28))     # test: 4                    train: 1
# input = np.reshape(data[7], (28, 28))     # test: 9                    train: 0
# input = np.reshape(data[8], (28, 28))     # test: 6                    train: 9
# input = np.reshape(data[9], (28, 28))     # test: 9 -> two parts       train: 8
# input = np.reshape(data[403], (28, 28))   # test: 8

# input[(input > 0) & (input < 0.5)] = 0.4
# input[input > 0] = 0.4

# input = mnist_test()
# input = mnist_test2()
# input = mnist_test3()
# input = mnist_test4()
# input = mnist_test5()
# input = mnist_test6()
# input = mnist_test7()
# input = mnist_test8()
# input = mnist_test9()
# input = mnist_test10()

num_bars = 0
if obstructed:
    #train
    num_bars = 3
    if test:
        #test
        num_bars = 4

print("num bars", num_bars)

def run_mnist_hisi(start, stop):
    print("From:", start, " to ", stop)

    stimuli = []
    stimuli2 = []
    for m, mnist in enumerate(data[start:stop]):
        print()
        print("################################")
        print("##########     "+str(m + start)+"      #########")
        print("################################")
        input = np.reshape(mnist, (28, 28))

        input[(input > 0) & (input < 0.5)] = 0.4

        images, boundaries = hisi(input, use_quadratic=False)

        all_img = []
        for i, img in enumerate(images):
            if show:
                plt.figure()
                plt.imshow(show_matrix(img, boundaries[i]))

            all_img.append(img)
            # scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save('img' + str(i) + '.jpg')
            # np.save('img'+str(num_obj), input)

        # # to save all the founded objects
        # output = np.zeros((28,28))
        # nb_tot_img = np.shape(all_img)[0]
        # nb_img = nb_tot_img - num_bars - 2
        # if nb_img > 0:
        #     for i in range(nb_img):
        #         output += all_img[2+i]
        # else:
        #     output = all_img[1]
        #
        # output[output < 0] = 0
        # stimuli.append(output)

        # to save the input stimuli plus the first found object
        # output = [all_img[0], all_img[-(num_bars + 1)]]
        # output = np.reshape(output2, (np.shape(output2)[0] * np.shape(output2)[1], np.shape(output2)[2]))
        # stimuli.append(output)

        # to save the last object
        output2 = all_img[-(num_bars + 1)]
        # scipy.misc.toimage(output, cmin=0.0, cmax=1.0` ).save('results/img'+str(m)+'.jpg')
        output2[output2 < 0] = 0
        stimuli2.append(output2)

        if show:
            plt.show()
        #
        # if m > 0 and m % 1000 == 0:
        #     np.save("results/mnist_fast_lami_train_" + str(m + start), stimuli)

        if save:
            # np.save("temp/mnist_HISI_"+str(start), stimuli)
            np.save("temp/mnist_HISI_"+str(start), stimuli2)

if __name__ == '__main__':
    startSimulationTime = datetime.datetime.now()

    if single_thread:
        # one thread
        run_mnist_hisi(0, 10)
    else:
        # multi processor
        # train: 55000 -> 11 cores / 5000 images
        # test: 10000 -> 16 cores / 625 images
        if test:
            npt = 625  # Number of images Per Thread
            num_cores = 16
        else:
            npt = 5000  # Number of images Per Thread
            num_cores = 11

        thread_list = []
        run_size = npt * num_cores

        for i in range(num_cores):
            batch_size = i * npt
            start = batch_size
            stop = start + npt
            p = Process(target=run_mnist_hisi, args=(start,stop))
            thread_list.append(p)

        for p in thread_list:
            p.start()

        for p in thread_list:
            p.join()


    endSimulationTime = datetime.datetime.now()
    print("Simulation time: " + str(endSimulationTime - startSimulationTime))