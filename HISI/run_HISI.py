import numpy as np
import matplotlib.pylab as plt
import scipy.misc
from HISI import *
import datetime
from multiprocessing import Process

#used for generating the pictures of the report
# use_quadratic = False
# input = square_3x3()
# input = exemple_pool() #need to remove the comment at line 389 in order to display the pooling boundaries
# input = grossberg() #set arguemnt to True for the reconstruction method line 344
# input = all_shades() #need to set again the pre-processing steps
# input = middle_square_3x3()
# input = two_squares()
# input = one_line()
# input = one_line_two_objects()
# input = seg_one_line_two_object()
# input = two_bars_one_seg_line()
# input = hori_line()
# input = hori_line_big()
# input = two_seg()

#Example where images are big enough so quadratic could be activated
use_quadratic = True
# input = hori_square_150()
# input = left_square_150()
# input = bottom_square_150()
# input = left_triangle_150()
# input = top_triangle_150()
# input = bottom_triangle_150()
# input = corner_150()
input = circle_150()
# input = dancer()

show = True

def run_hisi(input):
    images, boundaries = hisi(input, use_quadratic)

    if show:
        for i, img in enumerate(images):
            plt.figure()
            plt.imshow(show_matrix(img, boundaries[i]))
            scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save('results/img' + str(i) + '.jpg')

        plt.show()

if __name__ == '__main__':
    startSimulationTime = datetime.datetime.now()

    run_hisi(input)

    endSimulationTime = datetime.datetime.now()
    print("Simulation time: " + str(endSimulationTime - startSimulationTime))
