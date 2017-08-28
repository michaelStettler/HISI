import numpy as np
from input_test import *
from segmentation import *
from grouping import *
from show_matrix import *
from curve_completion import *


def fill_boundaries(boundaries, ind, filling_pixels, val_bound, thresh_bound):
    width = np.shape(boundaries)[2]
    height = np.shape(boundaries)[1]

    if ind[0] == 0:
        #top boundaries
        last_y = ind[1]
        for pix in filling_pixels:
            diff = pix[0] - last_y
            last_y = pix[0]
            if diff < 0:
                #means it's going up so we need to add a left bound
                if 0 <= pix[0] + 1 < height and 0 <= pix[1] - 1 < width:
                    if boundaries[2, pix[0] + 1, pix[1] - 1] <= thresh_bound:
                        boundaries[2, pix[0] + 1, pix[1] - 1] = val_bound
            else:
                #means it could go straight or down, so add a right bound
                if 0 <= pix[0] + 1 < height and 0 <= pix[1] + 1 < width:
                    if boundaries[3, pix[0] + 1, pix[1] + 1] <= thresh_bound:
                        boundaries[3, pix[0] + 1, pix[1] + 1] = val_bound

            if 0 <= pix[0] < height and 0 <= pix[1] < width:
                if boundaries[0, pix[0], pix[1]] <= thresh_bound:
                    boundaries[0, pix[0], pix[1]] = val_bound

    elif ind[0] == 1:
        #bottom boundaries
        last_y = ind[1]
        last_x = ind[2]
        for pix in filling_pixels:
            diff_y = pix[0] - last_y
            diff_x = pix[1] - last_x
            last_y = pix[0]
            last_x = pix[1]
            if diff_y > 0:
                # means it's going down so we need to add a right bound
                if 0 <= pix[0] - 1 < height and 0 <= pix[1] + 1 < width:
                    if boundaries[3, pix[0] - 1, pix[1] + 1] <= thresh_bound:
                        boundaries[3, pix[0] - 1, pix[1] + 1] = val_bound
            else:
                # means it could go straight or up
                if diff_x < 0:
                    #if it's going left, add left bound
                    if 0 <= pix[0] - 1 < height and 0 <= pix[1] - 1 < width:
                        if boundaries[2, pix[0] - 1, pix[1] - 1] <= thresh_bound:
                            boundaries[2, pix[0] - 1, pix[1] - 1] = val_bound

            if 0 <= pix[0] < height and 0 <= pix[1] < width:
                if boundaries[1, pix[0], pix[1]] <= thresh_bound:
                    boundaries[1, pix[0], pix[1]] = val_bound

    elif ind[0] == 2:
        #left boundaries
        last_x = ind[2]
        for pix in filling_pixels:
            diff = pix[1] - last_x
            last_x = pix[1]
            if diff < 0:
                # means it's going left so we need to add a bottom bound
                if 0 <= pix[0] + 1 < height and 0 <= pix[1] + 1 < width:
                    if boundaries[1, pix[0] + 1, pix[1] + 1] <= thresh_bound:
                        boundaries[1, pix[0] + 1, pix[1] + 1] = val_bound
            else:
                # means it could go up or right, so add a top bound
                if 0 <= pix[0] - 1 < height and 0 <= pix[1] + 1 < width:
                    if boundaries[0, pix[0] - 1, pix[1] + 1] <= thresh_bound:
                        boundaries[0, pix[0] - 1, pix[1] + 1] = val_bound

            if 0 <= pix[0] < height and 0 <= pix[1] < width:
                if boundaries[2, pix[0], pix[1]] <= thresh_bound:
                    boundaries[2, pix[0], pix[1]] = val_bound

    elif ind[0] == 3:
        # right boundaries
        last_x = ind[2]
        for pix in filling_pixels:
            diff = pix[1] - last_x
            last_x = pix[1]
            if diff > 0:
                # means it's going right so we need to add a top bound
                if 0 <= pix[0] - 1 < height and 0 <= pix[1] - 1 < width:
                    if boundaries[0, pix[0] - 1, pix[1] - 1] <= thresh_bound:
                        boundaries[0, pix[0] - 1, pix[1] - 1] = val_bound
            else:
                # means it could go down or left, so add a bottom bound
                if 0 <= pix[0] + 1 < height and 0 <= pix[1] - 1 < width:
                    if boundaries[1, pix[0] + 1, pix[1] - 1] <= thresh_bound:
                        boundaries[1, pix[0] + 1, pix[1] - 1] = val_bound

            if 0 <= pix[0] < height and 0 <= pix[1] < width:
                if boundaries[3, pix[0], pix[1]] <= thresh_bound:
                    boundaries[3, pix[0], pix[1]] = val_bound

    else:
        print("Problem with boundary in 'fill_boundaries' method in construct_bound.py")


def construct_boundaries(boundaries, inducers_pair, thresh_bound, max_induc_bound, use_quadratic, print_lab=False):
    #calculate filling curve
    if print_lab:
        print()
        print("inducers_pair:")
        for inducer in inducers_pair:
            print(inducer)
        print()

    for inducer in inducers_pair:
        ind0 = calculate_inducer(boundaries, inducer[0], thresh_bound, max_induc_bound, use_quadratic)
        p0 = [inducer[0][0], inducer[0][1], inducer[0][2]]

        ind1 = calculate_inducer(boundaries, inducer[1], thresh_bound, max_induc_bound, use_quadratic)
        p1 = [inducer[1][0], inducer[1][1], inducer[1][2]]
        filling_pixels, lines = find_completion_curve(p0, ind0, p1, ind1)

        if filling_pixels is not None:
            #we want a smaller value as we don't want the reconstruction to start from this place
            val_bound = min(boundaries[p0[0], p0[1], p0[2]] - 0.05, boundaries[p1[0], p1[1], p1[2]] - 0.05)

            if np.shape(lines)[0] == 1:
                fill_boundaries(boundaries, inducer[0], filling_pixels, val_bound, thresh_bound)
                #todo remove inner bound of segmentation

            elif np.shape(lines)[0] == 2:
                p = np.copy(p1)
                p[0] = lines[1]

                ind = [p0, p]
                for i, interpolation in enumerate(filling_pixels):
                    # interpolation[0, :]
                    fill_boundaries(boundaries, ind[i], interpolation, val_bound, thresh_bound)
            else:
                print("problem with the number of lines of the completion curve in counstruct_bound.py meth: construct_boundaries")

if __name__ == '__main__':
    thresh_bound = 0.3
    max_num_bounds = 1000
    min_num_bounds = 8
    max_induc_bound = 3  # maximum boundaries taking into account to calculate the curve


    # # ========= Test 1 - horizontal ==========#
    # input = segmented_hori_line()
    # boundaries = get_boundaries(input)
    #
    # boundaries[2, :, 4] = 0
    # boundaries[2, :, 5] = 0
    # boundaries[3, :, 4] = 0
    # boundaries[3, :, 5] = 0
    # boundaries[2, :, 11] = 0
    # boundaries[2, :, 12] = 0
    # boundaries[3, :, 11] = 0
    # boundaries[3, :, 12] = 0
    # boundaries[2, :, 18] = 0
    # boundaries[2, :, 19] = 0
    # boundaries[3, :, 18] = 0
    # boundaries[3, :, 19] = 0
    #
    # remaining_bound = np.copy(boundaries)
    # inducers_bound = np.zeros(np.shape(boundaries))
    # # use segment_remain_oject to get inducers
    # find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds)
    #
    # inducers_pair = group_inducers(boundaries, inducers_bound, thresh_bound, max_induc_bound)
    # construct_boundaries(boundaries, inducers_pair, thresh_bound, max_induc_bound, True)
    #
    # plt.figure()
    # plt.imshow(show_matrix(input, boundaries))
    #
    #
    # #========= Test 2 - vertical ==========#
    #
    # input = segmented_verti_line()
    # boundaries = get_boundaries(input)
    #
    # boundaries[0, 4, :] = 0
    # boundaries[0, 5, :] = 0
    # boundaries[1, 4, :] = 0
    # boundaries[1, 5, :] = 0
    # boundaries[0, 11, :] = 0
    # boundaries[0, 12, :] = 0
    # boundaries[1, 11, :] = 0
    # boundaries[1, 12, :] = 0
    # boundaries[0, 18, :] = 0
    # boundaries[0, 19, :] = 0
    # boundaries[1, 18, :] = 0
    # boundaries[1, 19, :] = 0
    #
    # remaining_bound = np.copy(boundaries)
    # inducers_bound = np.zeros(np.shape(boundaries))
    # # use segment_remain_oject to get inducers
    # find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds)
    #
    # inducers_pair = group_inducers(boundaries, inducers_bound, thresh_bound, max_induc_bound)
    # new_bound = construct_boundaries(boundaries, inducers_pair, thresh_bound, max_induc_bound)
    #
    # plt.figure()
    # plt.imshow(show_matrix(input, boundaries))

    # ========= Test 3 - left square (top -> bottom boundaries) ==========#
    max_induc_bound = 5

    input = left_square()
    boundaries = get_boundaries(input)

    boundaries[2, :, 5:] = 0
    boundaries[3, :, 5:] = 0

    remaining_bound = np.copy(boundaries)
    inducers_bound = np.zeros(np.shape(boundaries))
    # use segment_remain_oject to get inducers
    find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds)

    inducers_pair = group_inducers(boundaries, inducers_bound, thresh_bound, max_induc_bound)
    new_bound = construct_boundaries(boundaries, inducers_pair, thresh_bound, max_induc_bound)

    plt.figure()
    plt.imshow(show_matrix(input, boundaries))

    # ========= Test 4 - left triangle (top -> bottom boundaries) ==========#

    input = left_triangle()
    boundaries = get_boundaries(input)

    boundaries[2, :, 8:] = 0
    boundaries[3, :, 8:] = 0

    remaining_bound = np.copy(boundaries)
    inducers_bound = np.zeros(np.shape(boundaries))
    # use segment_remain_oject to get inducers
    find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds)

    inducers_pair = group_inducers(boundaries, inducers_bound, thresh_bound, max_induc_bound)
    new_bound = construct_boundaries(boundaries, inducers_pair, thresh_bound, max_induc_bound)

    plt.figure()
    plt.imshow(show_matrix(input, boundaries))

    # ========= Test 5 - top left corner ==========#

    input = top_left_corner()
    boundaries = get_boundaries(input)

    boundaries[0, 3:6, :] = 0
    boundaries[1, 3:6, :] = 0
    boundaries[3, 0:6, :] = 0

    remaining_bound = np.copy(boundaries)
    inducers_bound = np.zeros(np.shape(boundaries))
    # use segment_remain_oject to get inducers
    find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds)

    inducers_pair = group_inducers(boundaries, inducers_bound, thresh_bound, max_induc_bound)
    new_bound = construct_boundaries(boundaries, inducers_pair, thresh_bound, max_induc_bound)

    plt.figure()
    plt.imshow(show_matrix(input, boundaries))

    #
    #
    # # ========= Test 3 - triangle ==========#
    #
    # max_induc_bound = 8
    #
    # input = segmented_triangle()
    # boundaries = get_boundaries(input)
    #
    # # boundaries[0, 0:4, :] = 0
    # boundaries[0, 0:6, :] = 0
    # boundaries[1, 0:7, :] = 0
    #
    # remaining_bound = np.copy(boundaries)
    # inducers_bound = np.zeros(np.shape(boundaries))
    # # use segment_remain_oject to get inducers
    # find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds)
    #
    # inducers_pair = group_inducers(boundaries, inducers_bound, thresh_bound, max_induc_bound)
    # new_bound = construct_boundaries(boundaries, inducers_pair, thresh_bound, max_induc_bound)
    #
    # plt.figure()
    # plt.imshow(show_matrix(input, boundaries))

    plt.show()