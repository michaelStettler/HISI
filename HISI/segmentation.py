import numpy as np
from boundaries import *

def control_historic(new_loc, historic):
    for loc in historic:
        if loc[0] == new_loc[0] and loc[1] == new_loc[1] and loc[2] == new_loc[2]:
            return True

    return False



def has_been_segmented(input, loc, clockwise):
    """
    This method return if the path of a boundary end up in a segmented part, take into account that a segmented part of
    a picture is set to -1

    Parameters
    ----------
    input
    loc
    clockwise

    Returns
    -------

    """
    height = np.shape(input)[0]
    width = np.shape(input)[1]

    if clockwise:
        if loc[0] == 0 and loc[2] + 1 < width:
            if input[int(loc[1]), int(loc[2]) + 1] < 0:
                return True
        elif loc[0] == 1 and loc[2] - 1 >= 0:
            if input[int(loc[1]), int(loc[2]) - 1] < 0:
                return True
        elif loc[0] == 2 and loc[1] - 1 >= 0:
            if input[int(loc[1]) - 1, int(loc[2])] < 0:
                return True
        elif loc[1] + 1 < height:
            if input[int(loc[1]) + 1, int(loc[2])] < 0:
                return True
    else:
        if loc[0] == 0 and loc[2] - 1 >= 0:
            if input[int(loc[1]), int(loc[2]) - 1] < 0:
                return True
        elif loc[0] == 1 and loc[2] + 1 < width:
            if input[int(loc[1]), int(loc[2]) + 1] < 0:
                return True
        elif loc[0] == 2 and loc[1] + 1 < height:
            if input[int(loc[1]) + 1, int(loc[2])] < 0:
                return True
        elif loc[1] - 1 >= 0:
            if input[int(loc[1]) - 1, int(loc[2])] < 0:
                return True

    return False


def find_close_boundary(bound, init_bound, historic_bound, start_bound, clockwise, input, thresh_bound, num, min_num_bounds, max_num_bounds):
    is_close_bound = False
    found_border = False
    found_seg = False
    num_bounds = num
    last_bound = start_bound

    if num_bounds < max_num_bounds:
        #control if we have not reach the limit -> security against infinte loop
        if 0 < num_bounds and start_bound == init_bound:
            # the loop is complete and bigger than zero (to pass initial condition)
            is_close_bound = True
        else:
            #the loop is not complete
            #look for next boundary
            has_a_bound, new_loc, out_of_bound = find_next_boundaries(bound, start_bound, clockwise, thresh_bound)

            if has_a_bound:
                #has a bound so keep going on the path

                #first control if we are not enterring the infinite loop case
                is_in_historic = control_historic(new_loc, historic_bound)
                is_enterring_infinite_loop = num_bounds > min_num_bounds and is_in_historic
                if not is_enterring_infinite_loop:
                    num_bounds += 1
                    historic_bound.append(new_loc)
                    return find_close_boundary(bound, init_bound, historic_bound, new_loc, clockwise, input, thresh_bound,
                                               num_bounds, min_num_bounds, max_num_bounds)

            elif out_of_bound:
                #it's out of bound so we get out
                found_border = True
                last_bound = new_loc

            else:
                #no more bound
                if has_been_segmented(input, start_bound, clockwise):
                    #try if it's because segmentation
                    found_seg = True

    else:
        print("reach max bound number!", last_bound, "segmentation.py find_close_boundary()")

    return is_close_bound, found_border, found_seg, num_bounds, last_bound


def find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds, add_inducer=True):
    #find a starting bound (higher contrast)
    start_bound = find_start_bound(remaining_bound)
    # print()
    # print("start bound: ", start_bound, remaining_bound[start_bound[0], start_bound[1], start_bound[2]])

    has_an_object = False
    num_bounds = 0
    historic_bound = []

    #if the higher contrast is bigger than the threshold -> look for object
    if remaining_bound[int(start_bound[0]), int(start_bound[1]), int(start_bound[2])] > thresh_bound:
        clockwise = True

        is_close_bound, found_border, found_seg, num_bounds, last_bound = find_close_boundary(remaining_bound, start_bound,
                                                                                              historic_bound,
                                                                                              start_bound, clockwise,
                                                                                              input, thresh_bound,
                                                                                              num_bounds, min_num_bounds,
                                                                                              max_num_bounds)
        if is_close_bound and min_num_bounds < num_bounds:
            has_an_object = True

        elif found_border:
            #found a border, go to other direction
            clockwise = False
            start_bound = last_bound
            num_bounds = 0
            historic_bound = []
            historic_bound.append(start_bound)

            is_close_bound, found_border, found_seg, num_bounds, last_bound2 = find_close_boundary(remaining_bound, start_bound,
                                                                                                   historic_bound,
                                                                                              start_bound, clockwise,
                                                                                              input, thresh_bound,
                                                                                              num_bounds, min_num_bounds,
                                                                                                   max_num_bounds)


            if found_border:
                #mean we have reached two borders
                has_an_object = True
            elif found_seg:
                #mean we have reached one border and then a segmentation

                if add_inducer:
                    # if add_inducer is set, add the inducer
                    inducers_bound[int(last_bound2[0]), int(last_bound2[1]), int(last_bound2[2])] = remaining_bound[int(last_bound2[0]), int(last_bound2[1]), int(last_bound2[2])]

                remaining_bound[int(last_bound2[0]), int(last_bound2[1]), int(last_bound2[2])] = 0

                #look for next object
                return find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds,
                                        max_num_bounds, add_inducer)
            else:
                #empty border, remove the last border and start again to look for other possibilities
                remaining_bound[int(last_bound[0]), int(last_bound[1]), int(last_bound[2])] = 0

                return find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds,
                                        max_num_bounds, add_inducer)


        elif found_seg:
            if add_inducer:
                #if we can add the inducers, then try for other possibilities
                clockwise = False
                start_bound = last_bound
                num_bounds = 0
                historic_bound = []
                historic_bound.append(start_bound)

                is_close_bound, found_border, found_seg, num_bounds, last_bound2 = find_close_boundary(remaining_bound, start_bound,
                                                                                                       historic_bound,
                                                                                                  start_bound, clockwise,
                                                                                                  input, thresh_bound,
                                                                                                  num_bounds, min_num_bounds,
                                                                                                       max_num_bounds)

                if found_border:
                    # mean we have reached one seg and then a border
                    # add first inducer
                    inducers_bound[int(last_bound[0]), int(last_bound[1]), int(last_bound[2])] = remaining_bound[int(last_bound[0]), int(last_bound[1]), int(last_bound[2])]
                    remaining_bound[int(last_bound[0]), int(last_bound[1]), int(last_bound[2])] = 0

                    # look for next object
                    return find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds,
                                            max_num_bounds, add_inducer)

                if found_seg:

                    # mean we have reached two segmentation
                    #add the two inducers
                    inducers_bound[int(last_bound[0]), int(last_bound[1]), int(last_bound[2])] = remaining_bound[int(last_bound[0]), int(last_bound[1]), int(last_bound[2])]
                    inducers_bound[int(last_bound2[0]), int(last_bound2[1]), int(last_bound2[2])] = remaining_bound[int(last_bound2[0]), int(last_bound2[1]), int(last_bound2[2])]

                    remaining_bound[int(last_bound[0]), int(last_bound[1]), int(last_bound[2])] = 0
                    remaining_bound[int(last_bound2[0]), int(last_bound2[1]), int(last_bound2[2])] = 0
                    # look for next object
                    return find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds, add_inducer)
            else:
                #here we do not want to add the inducer
                #remove last boundary and start over to look for other possibilities
                remaining_bound[int(last_bound[0]), int(last_bound[1]), int(last_bound[2])] = 0
                return find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds, max_num_bounds, add_inducer)

        else:
            #no object found so delete the boundary path
            remaining_bound[int(last_bound[0]), int(last_bound[1]), int(last_bound[2])] = 0
            # print("start bound: ", start_bound, remaining_bound[start_bound[0], start_bound[1], start_bound[2]], "delete last bound:", last_bound)
            return find_next_object(input, remaining_bound, inducers_bound, thresh_bound, min_num_bounds,
                                        max_num_bounds, add_inducer)

    return has_an_object, start_bound

