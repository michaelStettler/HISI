import numpy as np
import matplotlib.pylab as plt
from input_test import *
from boundaries import *
from segmentation import *
from show_matrix import *
from inducers import *

def sort_inducers(boundaries, inducers_bound, thresh_bound):
    positions = np.where(inducers_bound > 0)
    num_inducers = np.shape(positions)[1]

    inducers = np.zeros((2, np.shape(boundaries)[0], np.shape(boundaries)[1], np.shape(boundaries)[2]))
    for i in range(num_inducers):
        loc_inducer = [positions[0][i], positions[1][i], positions[2][i]]
        clockwise = True
        has_a_bound, new_loc, out_of_bound = find_next_boundaries(boundaries, loc_inducer, clockwise, thresh_bound)
        ind = inducers_bound[loc_inducer[0], loc_inducer[1], loc_inducer[2]]
        if has_a_bound:
            inducers[1, loc_inducer[0], loc_inducer[1], loc_inducer[2]] = ind
        else:
            inducers[0, loc_inducer[0], loc_inducer[1], loc_inducer[2]] = ind

    return inducers

def calculate_same_bound_coeff(param1, param2):
    if param1 == param2:
        return 1.0
    elif (param1 == 0 and param2 == 3) or (param1 == 3 and param2 == 0):
        return 0.5
    elif (param1 == 0 and param2 == 1) or (param1 == 1 and param2 == 0):
        return 0.1
    elif (param1 == 1 and param2 == 2) or (param1 == 2 and param2 == 1):
        return 0.5
    elif (param1 == 2 and param2 == 3) or (param1 == 3 and param2 == 2):
        return 0.1
    else:
        print("problem with the bound! Grouping.py calculate_same_bound_coeff")
        print("param1: ", param1, " param2: ", param2)
        return 0.001


def calculate_linearity_coeff(lin1, lin2):
    """
    Calculate the coefficient for the type of the inducers, a linear inducer is more likely to be grouped with another
    linear inducer

    Parameters
    ----------
    lin1
    lin2

    Returns
    -------

    """
    if lin1 == lin2:
        return 1.0
    else:
        return 0.7


def calculate_distance(loc1, loc2):
    return np.power(np.e, 0.3 * np.sqrt(np.power(loc2[0] - loc1[0], 2) + np.power(loc2[1] - loc1[1], 2)))
    # return np.maximum(np.sqrt(np.power(loc2[0] - loc1[0], 2) + np.power(loc2[1] - loc1[1], 2)), 1)


def calculate_grad_intensity(grad1, grad2):
    return np.power(np.e, 2 * np.abs(grad2 - grad1))


def calculate_grad_angle(angle1, angle2):
    return 1 - np.abs(angle2 - angle1) / 180


def calculate_possibility(ind1, ind2, angle1):
    """
    This method is use to fine tune the cases of same boundaries
    Parameters
    ----------
    ind1
    ind2
    angle1

    Returns
    -------

    """
    if ind1[0] == 0 and ind2[0] == 0:
        if ind1[2] <= ind2[2]:
            #priority for boundaries more at the right as it is a top border
            if angle1 > 0:
                #if angle is bigger than zero, we give priority to upper boundary
                if ind2[1] <= ind1[1]:
                    return 1
                else:
                    return 0.4

            elif angle1 < 0:
                # if angle is smaller than zero, we give priority to bottom boundary
                if ind2[1] >= ind1[1]:
                    return 1
                else:
                    return 0.4

            else:
                #when angle is zero, there's no priority
                return 1
        else:
            return 0.01

    elif ind1[0] == 1 and ind2[0] == 1:
        if ind1[2] >= ind2[2]:
            #priority for boundaries more at the left as it is a bottom border
            if angle1 > 0:
                #if angle is bigger than zero, we give priority to bottom boundary
                if ind2[1] >= ind1[1]:
                    return 1
                else:
                    return 0.4

            elif angle1 < 0:
                # if angle is smaller than zero, we give priority to upper boundary
                if ind2[1] <= ind1[1]:
                    return 1
                else:
                    return 0.4

            else:
                # when angle is zero, there's no priority
                return 1

        else:
            return 0.01

    elif ind1[0] == 2 and ind2[0] == 2:
        if ind1[1] >= ind2[1]:
            #priority for boundaries more at the top (smaller in y) as it is a left border
            if angle1 > 0:
                #if angle is bigger than zero, we give priority to more right boundaries
                if ind2[2] >= ind1[2]:
                    return 1
                else:
                    return 0.4

            else:
                # if angle is smaller than zero, we give priority to more leff boundaries
                if ind2[2] <= ind1[2]:
                    return 1
                else:
                    return 0.4

        else:
            return 0.01

    elif ind1[0] == 3 and ind2[0] == 3:
        if ind1[1] <= ind2[1]:
            # priority for boundaries more at the bottom (higher in y) as it is a right border
            if angle1 > 0:
                # if angle is bigger than zero, we give priority to more left boundaries
                if ind2[2] <= ind1[2]:
                    return 1
                else:
                    return 0.4

            else:
                # if angle is smaller than zero, we give priority to more right boundaries
                if ind2[2] >= ind1[2]:
                    return 1
                else:
                    return 0.4

        else:
            return 0.01
    else:
        return 1


def calculate_scores(param_inducers1, param_inducers2, inducs1, inducs2, print_lab=False):
    scores = np.zeros((np.shape(param_inducers1)[0], np.shape(param_inducers2)[0]))
    positions = np.zeros((2, np.shape(param_inducers1)[0], np.shape(param_inducers2)[0], 3))

    for ind1 in range(np.shape(param_inducers1)[0]):
        for ind2 in range(np.shape(param_inducers2)[0]):
            c_bound = calculate_same_bound_coeff(param_inducers1[ind1][0][0], param_inducers2[ind2][0][0])
            c_linear = calculate_linearity_coeff(param_inducers1[ind1][1], param_inducers2[ind2][1])
            dist = calculate_distance(
                [param_inducers1[ind1][0][1],param_inducers1[ind1][0][2]],
                [param_inducers2[ind2][0][1],param_inducers2[ind2][0][2]])
            g_intensity = calculate_grad_intensity(inducs1[param_inducers1[ind1][0][0], param_inducers1[ind1][0][1], param_inducers1[ind1][0][2]],
                                                   inducs2[param_inducers2[ind2][0][0], param_inducers2[ind2][0][1], param_inducers2[ind2][0][2]])
            g_angle = calculate_grad_angle(param_inducers1[ind1][2], param_inducers2[ind2][2])
            g_possibility = calculate_possibility(param_inducers1[ind1][0], param_inducers2[ind2][0], param_inducers1[ind1][2])

            if print_lab:
                score = c_bound * c_linear * g_angle * g_possibility / (dist * g_intensity)
                print("induc1: ", param_inducers1[ind1][0], " inducs2: ", param_inducers2[ind2][0], " scores: ", score)
            scores[ind1, ind2] = c_bound * c_linear * g_angle * g_possibility / (dist * g_intensity)

            positions[0, ind1, ind2, :] = param_inducers1[ind1][0]
            positions[1, ind1, ind2, :] = param_inducers2[ind2][0]

    return scores, positions


def group_inducers(boundaries, inducers_bound, thresh_bound, max_induc_bound, use_quadratic, print_lab=False):
    #todo do some kind of normalization, threshold to pass test 2 when bars are closer ?
    inducers = sort_inducers(boundaries, inducers_bound, thresh_bound)
    grouped_induc = []

    while np.shape(np.where(inducers > 0))[1] > 0:

        param_clock = calculate_inducers(boundaries, inducers[0], thresh_bound, max_induc_bound, use_quadratic)
        param_counter_clock = calculate_inducers(boundaries, inducers[1], thresh_bound, max_induc_bound, use_quadratic)
        scores, positions = calculate_scores(param_clock, param_counter_clock, inducers[0], inducers[1], print_lab)

        if np.shape(np.where(scores > 0))[1] > 0:
            #greedy selection for the best "probability"
            best_1 = np.where(scores == np.amax(scores))[0][0]
            best_2 = np.where(scores == np.amax(scores))[1][0]

            if print_lab:
                print("scores")
                print(scores)
                print()

            pos_best_clock = positions[0, best_1, best_2]
            pos_best_counter = positions[1, best_1, best_2]

            grouped_induc.append([[int(pos_best_clock[0]), int(pos_best_clock[1]), int(pos_best_clock[2])],
                                  [int(pos_best_counter[0]), int(pos_best_counter[1]), int(pos_best_counter[2])]])
            inducers[0, int(pos_best_clock[0]), int(pos_best_clock[1]), int(pos_best_clock[2])] = 0.0
            inducers[1, int(pos_best_counter[0]), int(pos_best_counter[1]), int(pos_best_counter[2])] = 0.0

        else:
            inducers[inducers > 0] = 0

    if print_lab:
        print("grouped_induc")
        print(grouped_induc)

    return grouped_induc


if __name__ == '__main__':
    thresh_bound = 0.3
    max_num_bounds = 1000
    min_num_bounds = 8
    max_induc_bound = 5

    # # =================== Test 1 =====================#
    # print()
    # print("========= Test 1 ==========")
    # print()
    #
    # input = two_bar_inducers()
    # boundaries = get_boundaries(input)
    #
    # boundaries[2, :, 6] = 0
    # boundaries[2, :, 7] = 0
    # boundaries[3, :, 6] = 0
    # boundaries[3, :, 7] = 0
    #
    # remaining_bound = np.copy(boundaries)
    # inducers_bound = np.zeros(np.shape(boundaries))
    # #use segment_remain_oject to get inducers
    # find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds)
    #
    # group_inducers(boundaries, inducers_bound, thresh_bound, max_induc_bound, True)
    # plt.figure()
    # plt.imshow(show_matrix(input, inducers_bound))
    #
    # # =================== Test 2 =====================#
    # print()
    # print("========= Test 2 ==========")
    # print()
    #
    # input = two_half_bar_inducers()
    # boundaries = get_boundaries(input)
    #
    # boundaries[2, :, 6] = 0
    # boundaries[2, :, 7] = 0
    # boundaries[3, :, 6] = 0
    # boundaries[3, :, 7] = 0
    #
    # remaining_bound = np.copy(boundaries)
    # inducers_bound = np.zeros(np.shape(boundaries))
    # # use segment_remain_oject to get inducers
    # find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds)
    #
    # group_inducers(boundaries, inducers_bound, thresh_bound, max_induc_bound, True)
    # plt.figure()
    # plt.imshow(show_matrix(input, inducers_bound))

    # =================== Test 3 =====================#
    print()
    print("========= Test 3 ==========")
    print()

    input = test_mnist()
    boundaries = get_boundaries(input)

    boundaries[0, 3, :] = 0
    boundaries[0, 5, :] = 0
    boundaries[0, 6, :] = 0
    boundaries[0, 10, :] = 0
    boundaries[0, 11, :] = 0
    boundaries[1, 4, :] = 0
    boundaries[1, 5, :] = 0
    boundaries[1, 9, :] = 0
    boundaries[1, 10, :] = 0

    print(boundaries)
    remaining_bound = np.copy(boundaries)
    inducers_bound = np.zeros(np.shape(boundaries))
    # use segment_remain_oject to get inducers
    find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds)

    group_inducers(boundaries, inducers_bound, thresh_bound, max_induc_bound, True)
    plt.figure()
    plt.imshow(show_matrix(input, inducers_bound))

    plt.show()