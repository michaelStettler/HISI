import sys
import numpy as np
import os, os.path
import glob

output = []

output_name = "mnist_HISI"
test = True

if len(sys.argv) > 1:
    # arg order: output_name, test
    output_name = sys.argv[1]

    if sys.argv[2] == 'True':
        test = True
    else:
        test = False

# multi processor
# train: 55000 -> 11 cores / 5000 images
# test: 10000 -> 16 cores / 625 images
if test:
    npt = 625  # Number of images Per Thread
    num_cores = 16
else:
    npt = 5000  # Number of images Per Thread
    num_cores = 11

for i in range(num_cores):
    # condition_name = file.split("/")[1]
    print("condition name", i*npt)

    data = np.load("temp/mnist_HISI_"+str(i*npt)+".npy")
    print(np.shape(data))
    output.append(data)

print()
print("before:", np.shape(output))
output = np.reshape(output, (np.shape(output)[0] * np.shape(output)[1], np.shape(output)[2], np.shape(output)[3]))
print("after:", np.shape(output))
np.save("Stimuli/"+output_name, output)


# test
# npt = 625
# num_cores = 16
# for i in range(num_cores):
#     # condition_name = file.split("/")[1]
#     print("condition name", i*npt)
#
#     data = np.load("results/mnist_fast_lami2_"+str(i*npt)+".npy")
#     print(np.shape(data))
#     output.append(data)
#
# print()
# print("before:", np.shape(output))
# output = np.reshape(output, (np.shape(output)[0] * np.shape(output)[1], np.shape(output)[2], np.shape(output)[3]))
# print("after:", np.shape(output))
# np.save("../../../Stimuli/mnist_fast_lami2_4bar_test", output)

