import numpy as np
from scipy import signal
import math

def norm_matrix(matrix):
    for i in range(np.shape(matrix)[0]):
        if np.max(matrix[i]) <= 0:
            matrix[i] = matrix[i]
        else:
            matrix[i] /= np.max(matrix[i])

    return matrix


def pool_boundaries(boundaries, filter_size, coeff):
    """

    Parameters
    ----------
    boundaries : matrix conainting the weights of the boundaries
    filter_size : define the size of the pooling
    coeff : define the strength coefficient of the pooling

    Returns : new matrix of boundaries
    -------

    """
    pool = np.zeros(np.shape(boundaries))
    size_filters = np.arange(filter_size) + 1
    weight_pooling = [0.1, 0.08, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05] * coeff

    for i,size_filter in enumerate(size_filters):
        #vertical pooling
        pool[:, :-size_filter,:] += weight_pooling[i] * boundaries[:, size_filter:, :]
        pool[:, size_filter:, :] += weight_pooling[i] * boundaries[:, :-size_filter,:]
        #horizontal pooling
        pool[:, :, :-size_filter] += weight_pooling[i] * boundaries[:, :, size_filter:]
        pool[:, :, size_filter:] += weight_pooling[i] * boundaries[:, :, :-size_filter]

    pool[pool < 0] = 0

    return boundaries + pool


def pool_shade_boundaries(boundaries):
    pool = np.zeros(np.shape(boundaries))
    size_filters = [1, 2, 3]
    weight_pooling = [1.5, 1, 1]
    # weight_pooling = [.5, .7, .3]
    # size_filters = [1]
    # weight_pooling = [1]

    for k, size_filter in enumerate(size_filters):
        for i in range(size_filter, np.shape(boundaries)[1] - size_filter):
            for j in range(size_filter, np.shape(boundaries)[2] - size_filter):
                    pool[0, i, j] += weight_pooling[k] * (np.abs(
                        np.mean([boundaries[2, i - size_filter + 1, j - 1], boundaries[2, i + size_filter, j]]) -
                        np.mean([boundaries[2, i - size_filter + 1, j], boundaries[2, i + size_filter, j - 1]])
                    ))
                    pool[0, i, j] += weight_pooling[k] * (np.abs(
                        np.mean([boundaries[3, i - size_filter + 1, j], boundaries[3, i + size_filter, j + 1]]) -
                        np.mean([boundaries[3, i - size_filter + 1, j + 1], boundaries[3, i + size_filter, j]])
                    ))

                    pool[1, i, j] += weight_pooling[k] * (np.abs(
                        np.mean([boundaries[2, i - size_filter, j - 1], boundaries[2, i + size_filter - 1, j]]) -
                        np.mean([boundaries[2, i - size_filter, j], boundaries[2, i + size_filter - 1, j - 1]])
                    ))
                    pool[1, i, j] += weight_pooling[k] * (np.abs(
                        np.mean([boundaries[3, i - size_filter, j], boundaries[3, i + size_filter - 1, j + 1]]) -
                        np.mean([boundaries[3, i - size_filter, j + 1], boundaries[3, i + size_filter - 1, j]])
                    ))

                    pool[2, i, j] += weight_pooling[k] * (np.abs(
                        np.mean([boundaries[0, i - 1, j - size_filter + 1], boundaries[0, i, j + size_filter]]) -
                        np.mean([boundaries[0, i - 1, j + size_filter], boundaries[0, i, j - size_filter + 1]])
                    ))
                    pool[2, i, j] += weight_pooling[k] * (np.abs(
                        np.mean([boundaries[1, i, j - size_filter + 1], boundaries[1, i + 1, j + size_filter]]) -
                        np.mean([boundaries[1, i, j + size_filter], boundaries[1, i + 1, j - size_filter + 1]])
                    ))

                    pool[3, i, j] += weight_pooling[k] * (np.abs(
                        np.mean([boundaries[0, i - 1, j - size_filter + 1], boundaries[0, i, j + size_filter]]) -
                        np.mean([boundaries[0, i - 1, j + size_filter], boundaries[0, i, j - size_filter + 1]])
                    ))
                    pool[3, i, j] += weight_pooling[k] * (np.abs(
                        np.mean([boundaries[1, i, j - size_filter], boundaries[1, i + 1, j + size_filter - 1]]) -
                        np.mean([boundaries[1, i, j + size_filter - 1], boundaries[1, i + 1, j - size_filter]])
                    ))

    return boundaries + pool


def rem_iner_bound(input, boundaries, tresh):
    """

    Parameters
    ----------
    input : image input
    boundaries : matrix containing the weights of the boundaries
    tresh : threshold used as a sensibility coefficient, small threshold means high sensibility

    Returns the matrix of boundaries
    -------

    """
    pool = np.copy(boundaries)

    for i in range(np.shape(input)[0] - 1):
        for j in range(np.shape(input)[1] - 1):
            patch = input[i:i + 2, j:j + 2]
            min = np.min(patch)
            max = np.max(patch)
            diff = max-min
            mean = np.mean(patch)

            if 0 <= min:
                #remove boundaries of background and very similar colors
                if diff < 0.05:
                    boundaries[0, i, j:j + 2] -= pool[0, i, j:j + 2]
                    boundaries[1, i + 1, j:j + 2] -= pool[1, i + 1, j:j + 2]
                    boundaries[2, i:i + 2, j] -= pool[2, i:i + 2, j]
                    boundaries[3, i:i + 2, j + 1] -= pool[3, i:i + 2, j + 1]

                else:
                    if mean > 0.5 and diff < tresh:
                        boundaries[0, i, j:j + 2] -= pool[0, i, j:j + 2]
                        boundaries[1, i + 1, j:j + 2] -= pool[1, i + 1, j:j + 2]
                        boundaries[2, i:i + 2, j] -= pool[2, i:i + 2, j]
                        boundaries[3, i:i + 2, j + 1] -= pool[3, i:i + 2, j + 1]

    boundaries[boundaries < 0] = 0

    return boundaries


def rem_inner_seg_bound(input, boundaries):
    for i in range(np.shape(input)[0] - 1):
        for j in range(np.shape(input)[1] - 1):
            patch = input[i:i + 2, j:j + 2]
            neg = patch[patch < 0]
            if np.shape(neg)[0] == 4:
                boundaries[0, i, j:j + 2] = 0
                boundaries[1, i + 1, j:j + 2] = 0
                boundaries[2, i:i + 2, j] = 0
                boundaries[3, i:i + 2, j + 1] = 0

def choose_loc(x, y, dir):
    """
    Return the position of the next pixel in function of the direction one want to visit

    Parameters
    ----------
    x
    y
    dir

    Returns
    -------

    """
    if dir == 0:
        return [x-1,y]
    elif dir == 1:
        return [x+1,y]
    elif dir == 2:
        return [x, y-1]
    elif dir == 3:
        return [x, y+1]


def calculate_pixel(input, seg_img, boundaries, loc, thresh_bound):
    direction = []

    for dir in range(4):
        pos = choose_loc(loc[0], loc[1], dir)
        if 0 <= pos[0] < np.shape(input)[0] and 0 <= pos[1] < np.shape(input)[1]:
            if boundaries[dir, pos[0], pos[1]] < thresh_bound:

                if input[pos[0], pos[1]] > 0:
                    direction.append(dir)
                elif seg_img[pos[0], pos[1]] > 0:
                    direction.append(dir)

    for dir in direction:
        pos = choose_loc(loc[0], loc[1], dir)
        if input[pos[0],pos[1]] > 0:
            seg_img[loc[0], loc[1]] += (1 / np.shape(direction)[0]) * input[pos[0], pos[1]]
        else:
            seg_img[loc[0], loc[1]] += (1 / np.shape(direction)[0]) * seg_img[pos[0], pos[1]]


def fill_pixel(visited_pixel, input, bound, seg_img, thresh_bound):
    #fill pixels with the real pixel values from images
    for i in range(np.shape(input)[0]):
        for j in range(np.shape(input)[1]):
            if visited_pixel[i, j] == 1 and input[i, j] > 0:
                seg_img[i, j] = input[i, j]

    #fill pixels of segmented images
    #todo find a better way than this double loop
    #todo perhaps need to do this in the four direction for gradients colors?
    #top to down and left to right
    for i in range(np.shape(input)[0]):
        for j in range(np.shape(input)[1]):
            if visited_pixel[i, j] == 1 and input[i, j] < 0:
                calculate_pixel(input, seg_img, bound, [i, j], thresh_bound)

    #bottom -> top right -> left filling for remaining pixels
    for i in range(np.shape(input)[0] - 1, -1, -1):
        for j in range(np.shape(input)[1] - 1, -1, -1):
            if visited_pixel[i, j] == 1 and input[i, j] < 0 and seg_img[i, j] == 0:
                calculate_pixel(input, seg_img, bound, [i, j], thresh_bound)

    # # right -> left top -> bottom filling for remaining pixels
    # for i in range(np.shape(input)[1]):
    #     for j in range(np.shape(input)[0] - 1, -1, -1):
    #         if visited_pixel[j, i] == 1 and input[j, i] < 0 and seg_img[j, i] == 0:
    #             calculate_pixel(input, seg_img, bound, [j, i], thresh_bound)


def fill_shape(visited_pixel, input, boundaries, seg_img, seg_bound, loc, thresh_bound, num_iter):
    if 0 <= loc[0] < np.shape(visited_pixel)[0] and 0 <= loc[1] < np.shape(visited_pixel)[1]:
        visited_pixel[int(loc[0]), int(loc[1])] = 1
        num_iter += 1

        # dir = 0 go top
        # dir = 1 go down
        # dir = 2 go left
        # dir = 3 go right
        for dir in range(4):
            new_loc = choose_loc(loc[0], loc[1], dir)
            #verify if the next pixel is not out of range
            if 0 <= new_loc[0] < np.shape(visited_pixel)[0] and 0 <= new_loc[1] < np.shape(visited_pixel)[1]:
                if boundaries[int(dir), int(new_loc[0]), int(new_loc[1])] > thresh_bound:
                    seg_bound[int(dir), int(new_loc[0]), int(new_loc[1])] = boundaries[int(dir), int(new_loc[0]), int(new_loc[1])]
                else:
                    if not visited_pixel[int(new_loc[0]), int(new_loc[1])]:
                        fill_shape(visited_pixel, input, boundaries, seg_img, seg_bound, new_loc, thresh_bound, num_iter)



def define_contrast_edge_boundaries(boundary, positive):
    # this method make the assumption that the object is always in the center of the picture
    if positive:
        #control that there is boundaries: test cases
        if np.shape(np.nonzero(boundary)[0])[0] != 0:
            if boundary[np.nonzero(boundary)[0][0],np.nonzero(boundary)[1][0]] > 0:
                copy = np.copy(boundary)
                copy[copy <= 0] = 0
                return copy
            else:
                copy = np.copy(boundary)
                copy *= -1
                copy[copy <= 0] = 0
                return copy
        else:
            return boundary
    else:
        #control that there is boundaries: test cases
        if np.shape(np.nonzero(boundary)[0])[0] != 0:
            if boundary[np.nonzero(boundary)[0][0],np.nonzero(boundary)[1][0]] > 0:
                copy = np.copy(boundary)
                copy[copy <= 0] = 0
                return copy
            else:
                copy = np.copy(boundary)
                copy *= -1
                copy[copy <= 0] = 0
                return copy
        else:
            return boundary


def find_start_bound(boundaries):
    """
    This function take the maximum contrasted edges and return the position of the edge

    Parameters
    ----------
    boundaries

    Returns
    -------

    """
    max_contrast_bound = np.argmax(boundaries)
    shape = np.shape(boundaries)
    start_bound = [0, 0, 0]
    start_bound[0] = math.floor(max_contrast_bound / (shape[1] * shape[2]))
    rest = max_contrast_bound - start_bound[0] * shape[1] * shape[2]
    start_bound[1] = math.floor(rest / shape[2])
    start_bound[2] = rest - start_bound[1] * shape[2]

    return start_bound


def choose_next_bound(x, y, dir, bound):
    """
    Returns the next boundary position in function of the nature of the primary boundary (up, bottom, left, right) and
    take care of the direction

    Parameters
    ----------
    x
    y
    dir
    bound: matrix containing all the boundaries

    Returns
    -------

    """
    #top bound
    if bound == 0:
        #look for bottom right
        if dir == 0:
            return [3, x + 1, y + 1]
        #look for right top
        elif dir == 1:
            return [0, x, y + 1]
        # look for top right
        elif dir == 2:
            return [2, x, y]
        #look for right left
        elif dir == 3:
            return [2, x + 1, y - 1]
        #look for left top
        elif dir == 4:
            return [0, x, y - 1]
        #look for top left
        else:
            return [3, x, y]
    #down bound
    elif bound == 1:
        #look for top left
        if dir == 0:
            return [2, x - 1, y - 1]
        #look for left
        elif dir == 1:
            return [1, x, y - 1]
        #loof for bottom right
        elif dir == 2:
            return [3, x, y]
        #look for bottom right
        elif dir == 3:
            return [3, x - 1, y + 1]
        #look for down
        elif dir == 4:
            return [1, x, y + 1]
        #look for top right
        else:
            return [2, x, y]
    #left boundaries
    elif bound == 2:
        # look for top right
        if dir == 0:
            return [0, x - 1, y + 1]
        # look for left top
        elif dir == 1:
            return [2, x - 1, y]
        #loof for top left
        elif dir == 2:
            return [1, x, y]
        # look for bottom right
        elif dir == 3:
            return [1, x + 1, y + 1]
        # look for left bottom
        elif dir == 4:
            return [2, x + 1, y]
        # look for bottom left
        else:
            return [0, x, y]
    #right boundaries
    else:
        # look for bottom left
        if dir == 0:
            return [1, x + 1, y - 1]
        # look for right bottom
        elif dir == 1:
            return [3, x + 1, y]
        # look for bottom right
        elif dir == 2:
            return [0, x, y]
        # look for top left
        elif dir == 3:
            return [0, x - 1, y - 1]
        # look for right top
        elif dir == 4:
            return [3, x - 1, y]
        # look for top right
        else:
            return [1, x, y]


def find_next_boundaries(boundaries, loc, clockwise, thresh_bound, print_lab=False):
    out_of_bound = False
    # print()
    # print("location: ", loc)

    if clockwise:
        for dir in range(0, 3):
            new_loc = choose_next_bound(loc[1], loc[2], dir, loc[0])
            if 0 <= new_loc[1] < np.shape(boundaries)[1] and 0 <= new_loc[2] < np.shape(boundaries)[2]:

                if print_lab:
                    print("new loc", dir, new_loc, boundaries[new_loc[0], new_loc[1], new_loc[2]])

                if boundaries[int(new_loc[0]), int(new_loc[1]), int(new_loc[2])] > thresh_bound:
                    return True, new_loc, False
            else:
                out_of_bound = True

    else:
        for dir in range(3, 6):
            new_loc = choose_next_bound(loc[1], loc[2], dir, loc[0])
            if 0 <= new_loc[1] < np.shape(boundaries)[1] and 0 <= new_loc[2] < np.shape(boundaries)[2]:

                if print_lab:
                    print("new loc", dir, new_loc, boundaries[new_loc[0], new_loc[1], new_loc[2]])
                if boundaries[int(new_loc[0]), int(new_loc[1]), int(new_loc[2])] > thresh_bound:
                    return True, new_loc, False
            else:
                out_of_bound = True

    return False, loc, out_of_bound


def get_boundaries(input):
    image_height = np.shape(input)[0]
    image_width = np.shape(input)[1]

    # boundaries are vertical left = 0  vertical right = 1 and horizontal left = 2 horizontal right = 3
    boundaries = np.zeros((4, image_height, image_width))

    # set up boundaries filter
    # v1_hori_left = [[0,0,0],[1,-.2,-.8],[0,0,0]]
    # v1_hori_right = [[0,0,0],[-.8,-.2,1],[0,0,0]]
    v1_hori_left = [[0, 0, 0, 0, 0], [0.2, 1, -1.2, 0, 0], [0, 0, 0, 0, 0]]
    v1_hori_right = [[0, 0, 0, 0, 0], [0, 0, -1.2, 1, 0.2], [0, 0, 0, 0, 0]]
    v1_vert_top = np.transpose(v1_hori_left)
    v1_vert_down = np.transpose(v1_hori_right)

    # pass boundaries filter for each orientations
    filters = np.zeros((4, image_height, image_width))
    filters[0, :, :] = signal.convolve2d(input, v1_vert_top, boundary='symm', mode='same')
    filters[1, :, :] = signal.convolve2d(input, v1_vert_down, boundary='symm', mode='same')
    filters[2, :, :] = signal.convolve2d(input, v1_hori_left, boundary='symm', mode='same')
    filters[3, :, :] = signal.convolve2d(input, v1_hori_right, boundary='symm', mode='same')
    filters[filters < 0.00001] = 0

    boundaries[0, :, :] = define_contrast_edge_boundaries(filters[0, :, :], True)
    boundaries[1, :, :] = define_contrast_edge_boundaries(filters[1, :, :], False)
    boundaries[2, :, :] = define_contrast_edge_boundaries(filters[2, :, :], True)
    boundaries[3, :, :] = define_contrast_edge_boundaries(filters[3, :, :], False)

    return boundaries
