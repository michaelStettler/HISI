import numpy as np
from fit_curve import *
from input_test import *
from segmentation import *
from show_matrix import *


def add_border(loc, new_loc, clockwise, x, y, xdata, ydata):
    if clockwise:
        if loc[0] == 0:
            if new_loc[0] == 0:
                x += 1
                xdata.append(x)
                ydata.append(y)

            elif new_loc[0] == 2:
                x += 1
                y += 1
                xdata.append(x)
                ydata.append(y)

            # elif new_loc[0] == 3: #do nothing

        elif loc[0] == 1:
            if new_loc[0] == 1:
                x -= 1
                xdata.append(x)
                ydata.append(y)

            # elif new_loc[0] == 2: #do nothing

            elif new_loc[0] == 3:
                x -= 1
                y -= 1
                xdata.append(x)
                ydata.append(y)

        elif loc[0] == 2:
            # if new_loc[0] == 0: #do nothing

            if new_loc[0] == 1:
                x -= 1
                y += 1
                xdata.append(x)
                ydata.append(y)

            elif new_loc[0] == 2:
                y += 1
                xdata.append(x)
                ydata.append(y)

        elif loc[0] == 3:
            if new_loc[0] == 0:
                x += 1
                y -= 1
                xdata.append(x)
                ydata.append(y)

            # elif new_loc[0] == 1: #do nothing

            elif new_loc[0] == 3:
                y -= 1
                xdata.append(x)
                ydata.append(y)

        else:
            print("problem with boundary!!!")

    else: #counterclockwise
        if loc[0] == 0:
            if new_loc[0] == 0:
                x -= 1
                xdata.append(x)
                ydata.append(y)

            # elif new_loc[0] == 2: #do nothing

            elif new_loc[0] == 3:
                x -= 1
                y += 1
                xdata.append(x)
                ydata.append(y)

        elif loc[0] == 1:
            if new_loc[0] == 1:
                x += 1
                xdata.append(x)
                ydata.append(y)

            elif new_loc[0] == 2:
                x += 1
                y -= 1
                xdata.append(x)
                ydata.append(y)

            # elif new_loc[0] == 3: #do nothing

        elif loc[0] == 2:
            if new_loc[0] == 0:
                x -= 1
                y -= 1
                xdata.append(x)
                ydata.append(y)

            # elif new_loc[0] == 1: #do nothing

            elif new_loc[0] == 2:
                y -= 1
                xdata.append(x)
                ydata.append(y)

        elif loc[0] == 3:
            # new_loc[0] == 0: #do nothing
            if new_loc[0] == 1:
                x += 1
                y += 1
                xdata.append(x)
                ydata.append(y)

            elif new_loc[0] == 3:
                y += 1
                xdata.append(x)
                ydata.append(y)

        else:
            print("problem with boundary!!!")
    return x, y


def calculate_inducer(boundaries, init_loc, thresh_bound, max_induc_bound, use_quadratic, print_lab=False):
    """

    Parameters
    ----------
    boundaries
    init_loc
    thresh_bound
    max_induc_bound
    print_lab

    Returns is_linear, angle, radius, param, is_too_small
    -------

    """
    x = 0
    y = 0
    xdata = [x]
    ydata = [y]

    clockwise = True

    loc_inducer = init_loc
    has_a_bound, new_loc, out_of_bound = find_next_boundaries(boundaries, loc_inducer, clockwise, thresh_bound)
    if not has_a_bound:
        clockwise = False

    # for j in range(max_induc_bound): #old version
    num_borders = 0
    while math.fabs(x) < max_induc_bound and math.fabs(y) < max_induc_bound and num_borders < max_induc_bound * 1.5:
        num_borders += 1
        has_a_bound, new_loc, out_of_bound = find_next_boundaries(boundaries, loc_inducer, clockwise, thresh_bound)
        if has_a_bound:
            x, y = add_border(loc_inducer, new_loc, clockwise, x, y, xdata, ydata)
            loc_inducer = new_loc
        else:
            break


    if np.shape(xdata)[0] <= 1 or np.shape(ydata)[0] <= 1:
        #sometomes the inducers are formed by the pooling boundaries effect, so if they are not big enough the fin curve fails!!
        return True, 0.0, 0.0, [0], True

    else:
        is_linear, angle, radius, param = find_curve_param(xdata, ydata, use_quadratic, print_lab)

        if print_lab:
            print("xdata ", xdata)
            print("ydata ", ydata)
            print("params: ", is_linear, angle, radius, param)

        return is_linear, angle, radius, param, False


def calculate_inducers(boundaries, inducers_bound, thresh_bound, max_bound, use_quadratic, print_lab=False):
    positions = np.where(inducers_bound > 0)
    num_inducers = np.shape(positions)[1]

    mat = []
    if print_lab:
        print(num_inducers, " inducer(s) has been detected at pos: ", positions)

    for i in range(num_inducers):
        loc_inducer = [positions[0][i], positions[1][i], positions[2][i]]
        is_linear, angle, radius, param, is_too_small = calculate_inducer(boundaries, loc_inducer, thresh_bound, max_bound, use_quadratic, print_lab)

        if is_too_small:
            inducers_bound[positions[0][i], positions[1][i], positions[2][i]] = 0
        else:
            mat.append([loc_inducer, is_linear, angle, radius, param])

    return mat


if __name__ == '__main__':
    thresh_bound = 0.3
    max_num_bounds = 1000
    min_num_bounds = 8
    max_induc_bound = 10  # maximum boundaries taking into account to calculate the curve

    # =================== Test 1 =====================#
    print()
    print("========= Test 1 ==========")
    print()

    input = inducers_top_left_cross()
    boundaries = get_boundaries(input)

    boundaries[2, :, 12] = 0
    boundaries[3, :, 12] = 0

    remaining_bound = np.copy(boundaries)
    inducers_bound = np.zeros(np.shape(boundaries))
    #use segment_remain_oject to get inducers
    find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds)

    calculate_inducers(boundaries, inducers_bound, thresh_bound, max_induc_bound, True)

    plt.figure()
    plt.imshow(show_matrix(input, inducers_bound))

    # =================== Test 2 =====================#
    print()
    print("========= Test 2 ==========")
    print()

    input = inducers_down_left_cross()
    boundaries = get_boundaries(input)

    boundaries[2, :, 12] = 0
    boundaries[3, :, 12] = 0

    remaining_bound = np.copy(boundaries)
    inducers_bound = np.zeros(np.shape(boundaries))
    # use segment_remain_oject to get inducers
    find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds)

    calculate_inducers(boundaries, inducers_bound, thresh_bound, max_induc_bound, True)

    plt.figure()
    plt.imshow(show_matrix(input, inducers_bound))

    # =================== Test 3 =====================#
    print()
    print("========= Test 3 ==========")
    print()

    input = inducers_top_right_cross()
    boundaries = get_boundaries(input)

    boundaries[2, :, 0] = 0
    boundaries[3, :, 0] = 0

    remaining_bound = np.copy(boundaries)
    inducers_bound = np.zeros(np.shape(boundaries))
    # use segment_remain_oject to get inducers
    find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds)

    calculate_inducers(boundaries, inducers_bound, thresh_bound, max_induc_bound, True)

    plt.figure()
    plt.imshow(show_matrix(input, inducers_bound))

    # =================== Test 4 =====================#
    print()
    print("========= Test 4 ==========")
    print()

    input = inducers_down_right_cross()
    boundaries = get_boundaries(input)

    boundaries[2, :, 0] = 0
    boundaries[3, :, 0] = 0

    remaining_bound = np.copy(boundaries)
    inducers_bound = np.zeros(np.shape(boundaries))
    # use segment_remain_oject to get inducers
    find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds)

    calculate_inducers(boundaries, inducers_bound, thresh_bound, max_induc_bound, True)

    plt.figure()
    plt.imshow(show_matrix(input, inducers_bound))

    # =================== Test 5 =====================#
    print()
    print("========= Test 5 ==========")
    print()

    input = inducers_vert_top_left_cross()
    boundaries = get_boundaries(input)

    boundaries[0, 12, :] = 0
    boundaries[1, 12, :] = 0

    remaining_bound = np.copy(boundaries)
    inducers_bound = np.zeros(np.shape(boundaries))
    # use segment_remain_oject to get inducers
    find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds)

    calculate_inducers(boundaries, inducers_bound, thresh_bound, max_induc_bound, True)

    plt.figure()
    plt.imshow(show_matrix(input, inducers_bound))

    # =================== Test 6 =====================#
    print()
    print("========= Test 6 ==========")
    print()

    input = inducers_vert_top_right_cross()
    boundaries = get_boundaries(input)

    boundaries[0, 12, :] = 0
    boundaries[1, 12, :] = 0

    remaining_bound = np.copy(boundaries)
    inducers_bound = np.zeros(np.shape(boundaries))
    # use segment_remain_oject to get inducers
    find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds)

    calculate_inducers(boundaries, inducers_bound, thresh_bound, max_induc_bound, True)

    plt.figure()
    plt.imshow(show_matrix(input, inducers_bound))

    # =================== Test 7 =====================#
    print()
    print("========= Test 7 ==========")
    print()

    input = inducers_vert_down_left_cross()
    boundaries = get_boundaries(input)

    boundaries[0, 0, :] = 0
    boundaries[1, 0, :] = 0

    remaining_bound = np.copy(boundaries)
    inducers_bound = np.zeros(np.shape(boundaries))
    # use segment_remain_oject to get inducers
    find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds)

    calculate_inducers(boundaries, inducers_bound, thresh_bound, max_induc_bound, True)

    plt.figure()
    plt.imshow(show_matrix(input, inducers_bound))

    # =================== Test 8 =====================#
    print()
    print("========= Test 8 ==========")
    print()

    input = inducers_vert_down_right_cross()
    boundaries = get_boundaries(input)

    boundaries[0, 0, :] = 0
    boundaries[1, 0, :] = 0

    remaining_bound = np.copy(boundaries)
    inducers_bound = np.zeros(np.shape(boundaries))
    # use segment_remain_oject to get inducers
    find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds)

    calculate_inducers(boundaries, inducers_bound, thresh_bound, max_induc_bound, True)

    plt.figure()
    plt.imshow(show_matrix(input, inducers_bound))

    # =================== Test 9 =====================#
    print()
    print("========= Test 9 ==========")
    print()

    max_induc_bound = 5

    input = inducers_vert()
    boundaries = get_boundaries(input)

    boundaries[0, 12, :] = 0
    boundaries[1, 12, :] = 0

    remaining_bound = np.copy(boundaries)
    inducers_bound = np.zeros(np.shape(boundaries))
    # use segment_remain_oject to get inducers
    find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds)

    calculate_inducers(boundaries, inducers_bound, thresh_bound, max_induc_bound, True)

    plt.figure()
    plt.imshow(show_matrix(input, inducers_bound))

    plt.show()