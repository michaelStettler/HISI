import numpy as np
from scipy import signal

black = np.array([0,0.,1.,1.,1.,0.])
white = np.array([1,1.,0.,0.,0.,1.])
print(black)
print(white)

boundary = np.zeros((8, 6))

filter1 = [1,-1,0]
filter2 = [-1,1,0]
filter3 = [0,-1,1]
filter4 = [0,1,-1]

boundary[0,:] = signal.convolve(black,filter1, mode='same')
boundary[1,:] = signal.convolve(black,filter2, mode='same')
boundary[2,:] = signal.convolve(black,filter3, mode='same')
boundary[3,:] = signal.convolve(black,filter4, mode='same')

boundary[4,:] = signal.convolve(white,filter1, mode='same')
boundary[5,:] = signal.convolve(white,filter2, mode='same')
boundary[6,:] = signal.convolve(white,filter3, mode='same')
boundary[7,:] = signal.convolve(white,filter4, mode='same')
print("boundary")
print("black")
print(boundary[:4,:])
print("white")
print(boundary[4:,:])


print("test")
test1 = np.nonzero(boundary[0,:])
test2 = np.nonzero(boundary[4,:])

def define_contrast_edge_boundaries(boundary):
    # this method make the assumption that the object is always in the center of the picture
    if boundary[np.nonzero(boundary)[0][0]] > 0:
        copy = np.copy(boundary)
        copy[copy <= 0] = 0
        return copy
    else:
        copy = np.copy(boundary)
        copy *= -1
        copy[copy <= 0] = 0
        return copy


bound_filter0 = define_contrast_edge_boundaries(boundary[0,:])
bound_filter1 = define_contrast_edge_boundaries(boundary[4,:])


print("bound_filter")
print(bound_filter0)
print(bound_filter1)

# pos_filter0 = np.copy(boundary[0,:])
# pos_filter0[pos_filter0 < 0] = 0
# neg_filter0 = np.copy(boundary[0,:])
# neg_filter0[neg_filter0 >= 0] = 0
#
# pos_filter1 = np.copy(boundary[1,:])
# pos_filter1[pos_filter1 < 0] = 0
# neg_filter1 = np.copy(boundary[1,:])
# neg_filter1[neg_filter1 >= 1] = 0
#
# print("test")
# test = np.zeros((2, 6))
#
# test[0,:] += pos_filter0
# test[0,:] += pos_filter1
# test[1,1:] += neg_filter0[:-1]
# test[1,1:] += neg_filter1[:-1]
# print(test)
#
#
# pos_filter4 = np.copy(boundary[4,:])
# pos_filter4[pos_filter4 < 0] = 0
# neg_filter4 = np.copy(boundary[4,:])
# neg_filter4[neg_filter4 >= 0] = 0
#
# pos_filter5 = np.copy(boundary[5,:])
# pos_filter5[pos_filter5 < 0] = 0
# neg_filter5 = np.copy(boundary[5,:])
# neg_filter5[neg_filter5 >= 1] = 0
#
# print("test2")
# test2 = np.zeros((2, 6))
# test2[0,:] += pos_filter4
# test2[0,:] += pos_filter5
# test2[1,1:] += neg_filter4[:-1]
# test2[1,1:] += neg_filter5[:-1]
# print(test2)

#
# neg_filter1 = np.copy(boundary[1,:])
# neg_filter1[neg_filter1 >= 0] = 0
# print("neg_filter1")
# print(neg_filter1)
# pos_filter1 = np.copy(boundary[1,:])
# pos_filter1[pos_filter1 < 0] = 0
# print("pos_filter1")
# print(pos_filter1)

# test = np.zeros((2,6))
#
#
# res1 = np.zeros(6)
# res2 = np.zeros(6)
# res1[:-1] = boundary[0,:-1] - boundary[1,1:]
# res2[:-1] = boundary[4,:-1] - boundary[5,1:]
# print("res")
# print(res1)
# print(res2)




# neg_filter4 = np.copy(boundary[4,:])
# neg_filter4[neg_filter4 >= 0] = 0
# pos_filter4 = np.copy(boundary[4,:])
# pos_filter4[pos_filter4 < 0] = 0
#
# neg_filter5 = np.copy(boundary[5,:])
# neg_filter5[neg_filter5 >= 0] = 0
# pos_filter5 = np.copy(boundary[5,:])
# pos_filter5[pos_filter5 < 0] = 0
#
# res4 = pos_filter4 - neg_filter5
# res5 = neg_filter4 - pos_filter5
# print("res")
# print(res4)
# print(res5)