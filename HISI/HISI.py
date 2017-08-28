import numpy as np
import matplotlib.pylab as plt
import math
from scipy import misc
from show_matrix import *
from boundaries import *
from segmentation import *
from inducers import *
from construct_bound import *

import sys
sys.setrecursionlimit(10000)
np.set_printoptions(precision=2, linewidth=180)

def reconstruction(input, init_bound, pool_bound, start, inducers_bound, thresh_bound, max_induc_bound, min_num_bounds, max_num_bounds, num_objects, max_num_objects, output_img, output_bound, use_quadratic, pool, add_inducer=True):
    print()
    print(" ======== Reconstruction ========= ")
    print()
    print("Start with:", start)

    num_objects += 1
    finish = False
    #pool boundaries process, only when a object has been segmented
    pool_bound = np.copy(pool_bound)
    if pool:
        #todo set it as a ration of the image size
        pool_bound = pool_boundaries(pool_bound, 7, 12)
        pool_bound = rem_iner_bound(input, pool_bound, 0.9)
        thresh_bound /= 1.5

    #define new object matrixes and boundaries
    visited_pixel = np.zeros(np.shape(input))
    seg_img = np.zeros(np.shape(input))
    seg_bound = np.zeros(np.shape(init_bound))
    seg_bound[int(start[0]), int(start[1]), int(start[2])] = pool_bound[int(start[0]), int(start[1]), int(start[2])]

    #choose first pixel to be visited in function of the boudary
    if start[0] == 0:
        start[1] += 1
    elif start[0] == 1:
        start[1] -= 1
    elif start[0] == 2:
        start[2] += 1
    else:
        start[2] -= 1

    num_iter = 0
    fill_shape(visited_pixel, input, pool_bound, seg_img, seg_bound, [start[1], start[2]], thresh_bound, num_iter)
    fill_pixel(visited_pixel, input, seg_bound, seg_img, thresh_bound)
    print("shape reconstructed with:", num_iter)

    #looking for other objects in the picture
    remain_img = input
    remain_img[visited_pixel > 0] = -1

    remain_bound = init_bound
    remain_bound[seg_bound > 0] = 0
    remain_bound = rem_iner_bound(remain_img, remain_bound, 0.3)
    # rem_inner_seg_bound(remain_img, remain_bound)
    copy_remain_bound = np.copy(remain_bound)

    print()
    print("look for other object: (add_inducer)", add_inducer)
    found_object, start_bound = find_next_object(remain_img, copy_remain_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds, add_inducer)

    if found_object:
        print("found one more object")
        # Set pool to true for grossberg exemple
        if num_objects < max_num_objects:
            reconstruction(remain_img, init_bound, remain_bound, start_bound, inducers_bound, thresh_bound, max_induc_bound, min_num_bounds, max_num_bounds, num_objects, max_num_objects, output_img, output_bound, use_quadratic, pool=False, add_inducer=True)
        else:
            finish = True

    elif np.shape(np.where(inducers_bound > 0))[1] > 0:
        inducers_pair = group_inducers(remain_bound, inducers_bound, thresh_bound, max_induc_bound, use_quadratic)
        inducers_bound = np.zeros(np.shape(inducers_bound))
        print("No more object found, try with inducers")
        print("inducers_pair: ", inducers_pair)
        construct_boundaries(remain_bound, inducers_pair, thresh_bound, max_induc_bound, use_quadratic)
        print("Boundaries constructed")

        copy_remain_bound2 = np.copy(remain_bound)
        # inducers_bound = np.zeros(np.shape(inducers_bound))

        found_object, start_bound = find_next_object(remain_img, copy_remain_bound2, inducers_bound, thresh_bound,
                                                     min_num_bounds, max_num_bounds, add_inducer=False)

        if found_object:
            print("A new object has been found!")
            # inducers_bound = np.zeros(np.shape(inducers_bound))
            if num_objects < max_num_objects:
                reconstruction(remain_img, init_bound, remain_bound, start_bound, inducers_bound, thresh_bound, max_induc_bound, min_num_bounds, max_num_bounds, num_objects, max_num_objects, output_img, output_bound, use_quadratic, pool=False, add_inducer=False)
            else:
                finish = True
        else:
            print("no more object found")
            finish = True

    else:
        print("no more objects and no more inducers -> done")
        print()
        finish = True

    if finish:
        output_img.append(remain_img)
        output_bound.append(remain_bound)

    output_img.append(seg_img)
    output_bound.append(seg_bound)


def segmentation(input, boundaries, thresh_bound, max_induc_bound, min_num_bounds, max_num_bounds, num_objects, max_num_objects, output_img, output_bound, use_quadratic):
    num_objects += 1
    boundaries = norm_matrix(boundaries)
    start_bound = find_start_bound(boundaries)

    pool_bound = np.copy(boundaries)

    #pre processing of the boundaries
    #todo set the size as a matter of the image size
    # pool_bound = pool_boundaries(pool_bound, 3, 10) # V1
    # pool_bound = pool_boundaries(pool_bound, 8, 7) # V2

    # pooling exemple display
    # plt.figure()
    # plt.imshow(show_matrix(input, pool_bound))

    # pool_bound = pool_shade_boundaries(pool_bound)
    # pool_bound = rem_iner_bound(input, pool_bound, 0.4) #for shades set to 0.9 #todo, something to take care of this shading parameter
    pool_bound = norm_matrix(pool_bound)

    #get argmax boundary and its index
    #here choose the method of which object first, for now the max boundaries is taken and the first one of the array
    #todo method to find an object that uses other approach than the highest contrast ?

    print("max boundary: ", boundaries[int(start_bound[0]),int(start_bound[1]),int(start_bound[2])], "in (dept, row, col): (",
          start_bound[0], ",", start_bound[1], ",", start_bound[2], ")")

    num_bounds = 0
    inducers_bound = np.zeros(np.shape(boundaries))
    found_object, start_bound = find_next_object(input, pool_bound, inducers_bound, thresh_bound,
                                                 min_num_bounds, max_num_bounds, add_inducer=True)

    # if is_close_bound:
    if found_object:
        print("A shape has been found!")
        print("Size: ", num_bounds)
        print()
        reconstruction(input, boundaries, pool_bound, start_bound, inducers_bound, thresh_bound, max_induc_bound, min_num_bounds, max_num_bounds, num_objects, max_num_objects, output_img, output_bound, use_quadratic, pool=False, add_inducer=True)
    else:
        print("No close boundary found!!!")

def hisi(input, use_quadratic=True):
    print()
    print("====== HISI ======")
    print()

    #paramaters settings
    num_objects = 0
    max_num_objects = 10
    max_num_bounds = 1000  # repesent the maximum size of an object (security)
    min_num_bounds = 7 # represent the minimum border of boundaries before being considred as an object, in order to avoid "noisy pixel" 4 = 1 pixel, 6 = 2 pixels
    thresh_bound = 0.25
    # todo change max bound as a function of the image size ?
    max_induc_bound = 3
    if np.shape(input)[0] > 100:
        max_induc_bound = 14

    print("max induc bound: ", max_induc_bound)

    boundaries = get_boundaries(input)


    output_img = []
    output_bound = []
    output_img.append(np.copy(input))
    output_bound.append(np.copy(boundaries))

    # plt.figure()
    # plt.imshow(show_matrix(input, boundaries))
    # plt.show()

    segmentation(input, boundaries, thresh_bound, max_induc_bound, min_num_bounds, max_num_bounds, num_objects, max_num_objects, output_img, output_bound, use_quadratic)

    print()
    print("====== Finish ======")

    return output_img, output_bound
