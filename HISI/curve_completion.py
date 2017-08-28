import numpy as np
import matplotlib.pyplot as plt
import math
from euler_arc_spline import *

# def find_completion_curve(ind1, ind2, print_lab=False):
#     dist = np.sqrt((np.power(ind1[-1, 0] - ind2[0, 0], 2) + np.power(ind1[-1, 1] - ind2[0, 1], 2)))
#     num_points = math.ceil(10 * dist)
#
#     x = np.concatenate((ind1[:, 0],ind2[:, 0]), axis=0)
#     y = np.concatenate((ind1[:, 1],ind2[:, 1]), axis=0)
#
#     f = interp1d(x, y, kind='cubic')
#     xnew = np.linspace(0, ind2[-1, 0], num=num_points, endpoint=True)
#
#     interpolation = []
#     for i in range(int(num_points / 5) - 1):
#         interpolation.append([int(round(i * 5 / dist)), int(round(f(xnew)[i*5]))])
#
#     if print_lab:
#         plt.figure()
#         plt.plot(x, y, 'o', xnew, f(xnew), '--')
#         plt.legend(['data', 'cubic'], loc='best')
#
#     return interpolation


def find_completion_curve(p0, ind0, p1, ind1, print_lab=False):
    mult_factor = 5
    #todo better if we control the distance between pixel and ensure to close the gap than just a factor -> think about the slope parameter

    if print_lab:
        print()
        print(" =================== ")
        print()
        print("p0:", p0, " ind0:", ind0)
        print("p1:", p1, " ind1:", ind1)

    interpolation = []
    if ind0[0] and ind1[0]:
        #if both inducers are from straight line -> reconstruct without curvature
        # ===== top boundaries cases =====#
        if p0[0] == 0:
            if p1[0] == 0:
                if ind0[1] == ind1[1]:
                    #has the same angle
                    if p1[2] != p0[2]:
                        a = (p1[1] - p0[1]) / (p1[2] - p0[2])
                        lines = [0]
                        for x in range(int(math.ceil(math.fabs(p1[2] - p0[2]))) * mult_factor + 1):
                            # top boundaries
                            interpolation.append([int(round(a * x / mult_factor + p0[1])), int(p0[2] + x / mult_factor)])
                else:
                    #one line with different angle -> the middle point has to be calculated

                    #parameters first curve
                    a0 = ind0[3][0]
                    b0 = p0[1] - a0 * p0[2]
                    #parameters second curve
                    a1 = ind1[3][0]
                    b1 = p1[1] - a1 * p1[2]

                    if not math.isinf(b0) and not math.isinf(b1) and math.fabs(a1 - a0) > 0.01:
                        x_m = int(round((b1 - b0)/(a0 - a1)))
                        lines = [0] #there's only one final line between the two inducers
                        for x in range(p0[2] * mult_factor, x_m * mult_factor):
                            interpolation.append([int(round(a0 * x / mult_factor + b0)), int(round(x / mult_factor))])

                        for x in range(x_m * mult_factor, p1[2] * mult_factor + 1):
                            interpolation.append([int(round(a1 * x / mult_factor + b1)), int(round(x / mult_factor))])

            elif p1[0] == 1:
                #case where the top and bottom boundaries join to close the shape (square)
                if ind0[1] - ind1[1] < 0:
                    #the lines will meet
                    # parameters first curve
                    a0 = -ind0[3][0]
                    b0 = p0[1] - a0 * p0[2]
                    # parameters second curve
                    a1 = -ind1[3][0]
                    b1 = p1[1] - a1 * p1[2]

                    if not math.isinf(b0) and not math.isinf(b1) and math.fabs(a1 - a0) > 0.01:
                        x_m = (b0 - b1) / (a1 - a0)
                        y_m = int(round(a0 * x_m + b0))

                        line1 = []
                        line2 = []
                        lines = [0, 3]
                        if -0.01 < a0 < 0.01:
                            line1.append([p0[1], p0[2]])
                        else:
                            for x in range(p0[1] * mult_factor, y_m * mult_factor):
                                line1.append([int(round(x / mult_factor)), int(round((x / mult_factor - b0) / a0))])

                        if -0.01 < a1 < 0.01:
                            line2.append([p1[1], p1[2]])
                        else:
                            for x in range(y_m * mult_factor - 1, p1[1] * mult_factor + 1):
                                line2.append([int(round(x / mult_factor)), int(round((x / mult_factor - b1) / a1))])

                        interpolation = [line1, line2]

                else:
                    #the lines will never meet so we just go from one point to the other one
                    if p1[1] != p0[1]:
                        a = (p1[2] - p0[2]) / (p1[1] - p0[1])

                        line2 = []
                        lines = [0, 3]
                        for x in range(int(math.ceil(math.fabs(p1[1] - p0[1]))) * mult_factor + 1):
                            # top boundaries
                            line2.append([int(round(p0[1] + x / mult_factor)), int(round(a * x / mult_factor + p0[2])) + 1])

                        interpolation = [[[p0[1], p0[2]]], line2]

            elif p1[0] == 3:
                if ind0[1] - math.fabs(ind1[1]) > 0:
                    print("will never meet")

                else:
                    # the two lines will meet
                    # parameters first curve
                    a0 = ind0[3][0]
                    b0 = p0[1] - a0 * p0[2]
                    if not math.isinf(b0):
                        add_lines = True
                        # parameters second curve
                        if ind0 == 0 and ind1[1] == 90:
                            a1 = 0
                            x_m = p1[2]
                            y_m = p0[1]
                        elif ind1[1] == 90:
                            a1 = 0
                            x_m = p1[2]
                            y_m = a0 * x_m + b0
                        else:
                            a1 = -ind1[3][0]
                            b1 = p1[1] - a1 * p1[2]

                            if int(a1) != int(a0):
                                x_m = (b0 - b1) / (a1 - a0)
                                y_m = int(round(a0 * x_m + b0))
                                x_m = int(round(x_m))
                            else:
                                add_lines = False

                        if add_lines:
                            #todo careful if the shape is to small and they actually meet outside
                            line1 = []
                            line2 = []
                            lines = [0, 3]
                            for x in range(p0[2] * mult_factor, x_m * mult_factor + 1):
                                line1.append([int(round(a0 * x / mult_factor + b0)), int(round(x / mult_factor))])

                            if p1[1] - y_m <= 0:
                                line2.append([p1[1], p1[2]])
                            else:
                                for x in range(int(y_m * mult_factor), int(p1[1] * mult_factor + 1)):
                                    line2.append([int(round(x / mult_factor)), int(round(a1 * x / mult_factor + x_m))])

                            interpolation = [line1, line2]
            else:
                print("problem with boundaries in curve_completion.py meth: find_completion_curve")

        # ===== bottom boundaries cases =====#
        elif p0[0] == 1 and p1[0] == 1:
            if p1[2] != p0[2]:
                a = (p1[1] - p0[1]) / (p1[2] - p0[2])
                lines = [1]
                for x in range(int(math.ceil(math.fabs(p1[2] - p0[2]))) * mult_factor + 1):
                    # bottom boundaries -> remember clockwise so have to go to the other side
                    interpolation.append([int(round(a * (-x / mult_factor) + p0[1])), int(p0[2] - x / mult_factor)])

        # ===== left boundaries cases =====#
        elif p0[0] == 2:
            if p1[0] == 0:
                print("todo case 2")
                lines = [2]
                interpolation = []
            elif p1[0] == 2:
                if p1[1] != p0[1]:
                    a = (p1[2] - p0[2]) / (p1[1] - p0[1])
                    lines = [2]
                    for x in range(int(math.ceil(math.fabs(p1[1] - p0[1]))) * mult_factor + 1):
                        interpolation.append([int(p0[1] - x / mult_factor), int(round(a * -x / mult_factor + p0[2]))])

            elif p1[0] == 3:
                if ind0[1] < 0:
                    ind0 = list(ind0)
                    ind0[1] += 180

                if ind1[1] < 0:
                    ind1 = list(ind1)
                    ind1[1] += 180

                if p1[2] < p0[2]:
                    #inner bound
                    #todo cases of if the line could meet or not, now it's just completion between the two points
                    if p1[2] != p0[2]:
                        a = (p1[1] - p0[1]) / (p1[2] - p0[2])

                        line2 = []
                        lines = [2, 1]
                        for x in range(int(math.ceil(math.fabs(p1[2] - p0[2]))) * mult_factor + 1):
                            line2.append(
                                [int(a * x / mult_factor + p0[1]), int(p0[2] - x / mult_factor)])

                        interpolation = [[[p0[1] - 1, p0[2]]], line2]
                else:
                    #external boundaries
                    if ind0[1] < ind1[1]:
                        #the line will meet

                        line1 = []
                        line2 = []
                        lines = [0, 3]

                        if ind0[1] == 90:
                            a1 = -ind1[3][0]
                            b1 = p1[1] - a1 * p1[2]

                            x_m = p0[2]
                            y_m = int(a1 * x_m + b1)

                            for x in range((p0[2] - y_m) * mult_factor + 1):
                                line1.append([max(int(- x / mult_factor + x_m), 1), max(p0[2], 1)])

                            for x in range(int((p1[2] - x_m) * mult_factor + 1)):
                                line2.append([max(int(a1 * (x_m + x / mult_factor) + b1), 1), max(int(x / mult_factor + x_m), 1)])

                        elif ind1[1] == 90:
                            a0 = -ind0[3][0]
                            b0 = p0[1] - a0 * p0[2]

                            x_m = p1[2]
                            y_m = int(a0 * x_m + b0)

                            for x in range((x_m - p0[2]) * mult_factor + 1):
                                line1.append([max(int(a0 * (p0[2] + x / mult_factor) + b0), 1), int(x / mult_factor) + p0[2]])

                            for x in range(int((p1[1] - y_m) * mult_factor + 1)):
                                line2.append([max(int(x / mult_factor + y_m), 0), p1[2] - 1])

                        elif ind1[1] < 90:
                            print("todo case where it meet after the line")
                            a0 = -ind0[3][0]
                            b0 = p0[1] - a0 * p0[2]

                            x_m = p1[2]
                            y_m = int(a0 * x_m + b0)

                            for x in range((x_m - p0[2]) * mult_factor + 1):
                                line1.append(
                                    [max(int(a0 * (p0[2] + x / mult_factor) + b0), 1), int(x / mult_factor) + p0[2]])

                            for x in range(int((p1[1] - y_m) * mult_factor + 1)):
                                line2.append([max(int(x / mult_factor + y_m), 0), p1[2] - 1])

                        else:
                            #the two curve will meet in the middle
                            a0 = -ind0[3][0]
                            b0 = p0[1] - a0 * p0[2]

                            a1 = -ind1[3][0]
                            b1 = p1[1] - a1 * p1[2]

                            if math.fabs(a1 - a0) > 0.01:
                                x_m = int((b0 - b1) / (a1 - a0))

                                for x in range((x_m - p0[2]) * mult_factor + 1):
                                    line1.append([max(int(a0 * (p0[2] + x / mult_factor) + b0 + 1), 1), max(int(x / mult_factor) + p0[2], 1)])

                                for x in range(int((p1[2] - x_m) * mult_factor + 1)):
                                    line2.append([max(int(a1 * (x_m + x / mult_factor) + b1 + 1), 1), max(int(x / mult_factor + x_m), 1)])

                        interpolation = [line1, line2]

                    else:
                        # the lines will never meet so we just go from one point to the other one
                        if p1[2] != p0[2]:
                            a = (p1[1] - p0[1]) / (p1[2] - p0[2])

                            line2 = []
                            lines = [2, 0]
                            for x in range(int(math.ceil(math.fabs(p1[2] - p0[2]))) * mult_factor + 1):
                                # +1 because have to add the difference for the down boundaries
                                line2.append(
                                    [int(-a * x / mult_factor + p0[1]) - 1, int(p0[2] + x / mult_factor)])

                            interpolation = [[[p0[1] - 1, p0[2]]], line2]

                        else:
                            #possibility that the two inducers are at the same positions because of the indexes for left and right
                            #so just construct two points
                            lines = [2, 0]
                            interpolation = [[[p0[1] - 1, p0[2] - 1]], [[p1[1] - 1, p1[2]]]]


        # ===== right boundaries cases =====#
        elif p0[0] == 3:
            if p1[0] == 0:
                print("todo case 3")
                lines = [3]
                interpolation = []
            elif p1[0] == 2:
                #second boundary is a left one
                if ind0[1] < 0:
                    ind0 = list(ind0)
                    ind0[1] += 180

                if ind1[1] < 0:
                    ind1 = list(ind1)
                    ind1[1] += 180

                if p0[2] < p1[2]:
                    #inner boundaries -> just connect them for now #todo if line can meet or not =)
                    if p1[2] != p0[2]:
                        a = (p1[1] - p0[1]) / (p1[2] - p0[2])

                        line2 = []
                        lines = [3, 0]
                        # line goes to right (inner bound)
                        for x in range(int(math.ceil(math.fabs(p1[2] - p0[2]))) * mult_factor + 1):
                            line2.append(
                                [int(a * x / mult_factor + p0[1]), int(p0[2] + x / mult_factor)])
                        interpolation = [[[p0[1] + 1, p0[2]]], line2]

                else:
                    #external case
                    if ind1[1] > ind0[1]:
                        # the lines will meet
                        if ind0[1] == 90:
                            # parameters second curve
                            a1 = -ind1[3][0]
                            b1 = p1[1] - a1 * p1[2]

                            x_m = p0[2]
                            y_m = int(a1 * x_m + b1)

                            line1 = []
                            line2 = []
                            lines = [3, 1]

                            for x in range((y_m - p0[1]) * mult_factor + 1):
                                line1.append([int(x / mult_factor + p0[1]), x_m])

                            for x in range((x_m - p1[2]) * mult_factor + 1):
                                line2.append([int(a1 * (y_m - x) / mult_factor + p1[1]),
                                              int(a1 * (x_m - x) / mult_factor + p1[2] + 1)])

                            interpolation = [line1, line2]

                        elif ind1[1] == 90:
                            # 90 degree case

                            # parameters first curve
                            a0 = -ind0[3][0]
                            b0 = p0[1] - a0 * p0[2]

                            x_m = p1[2]
                            y_m = int(a0 * x_m + b0)

                            line1 = []
                            line2 = []
                            lines = [3, 1]

                            for x in range((p0[2] - x_m) * mult_factor + 1):
                                line1.append([max(int(a0 * (p0[2] - x / mult_factor) + b0), 1),
                                              max(int(p0[2] - x / mult_factor), 1)])

                            for x in range((y_m - p1[1]) * mult_factor + 1):
                                line2.append([int(y_m - x / mult_factor), p1[2] + 1])

                            interpolation = [line1, line2]

                        elif ind1[1] < 90:
                            #line meet before the two point
                            # now just combine the 2 points todo
                            print("take care of the case where ind1 < 90", p0, p1)
                            if p1[2] != p0[2]:
                                a = (p1[1] - p0[1]) / (p1[2] - p0[2])

                                line2 = []
                                lines = [3, 1]
                                for x in range(int(math.ceil(math.fabs(p1[2] - p0[2]))) * mult_factor + 1):
                                    # +1 because have to add the difference for the down boundaries
                                    line2.append([int(round(-a * x / mult_factor + p0[1]) + 1), int(round(p0[2] - x / mult_factor))])
                                interpolation = [[[p0[1] + 1, p0[2]]], line2]

                        elif ind0[1] > 90:
                            #need to do it, now only like ind0[1] == 90
                            # parameters second curve
                            a1 = -ind1[3][0]
                            b1 = p1[1] - a1 * p1[2]

                            x_m = p0[2]
                            y_m = int(a1 * x_m + b1)

                            line1 = []
                            line2 = []
                            lines = [3, 1]

                            for x in range((y_m - p0[1]) * mult_factor + 1):
                                line1.append([int(x / mult_factor + p0[1]), x_m])

                            for x in range((x_m - p1[2]) * mult_factor + 1):
                                line2.append([int(a1 * (y_m - x) / mult_factor + p1[1]),
                                              int(a1 * (x_m - x) / mult_factor + p1[2] + 1)])

                            interpolation = [line1, line2]

                        elif ind1[1] > 90:
                            #line meet between of the two points
                            # parameters first curve
                            a0 = -ind0[3][0]
                            b0 = p0[1] - a0 * p0[2]
                            # parameters second curve
                            a1 = -ind1[3][0]
                            b1 = p1[1] - a1 * p1[2]

                            if not math.isinf(b0) and not math.isinf(b1) and math.fabs(a1 - a0) > 0.01:  # todo take care of 90 degree cases
                                x_m = int((b0 - b1) / (a1 - a0))

                                line1 = []
                                line2 = []
                                lines = [3, 2]
                                for x in range((p0[2] - x_m) * mult_factor + 1):
                                    line1.append([max(int(a0 * (p0[2] - x / mult_factor) + b0), 1),
                                                  max(int(p0[2] - x / mult_factor), 1)])

                                for x in range((x_m - p1[2]) * mult_factor + 2):
                                    # need an extra pixel to close the shape..
                                    line2.append([max(int(a1 * (x_m - (x - 1) / mult_factor) + b1), 1),
                                                  max(int(x_m - (x - 1) / mult_factor), 1)])
                                interpolation = [line1, line2]


                        else:
                            #the line will meet at the exterior
                            #just connect the two points
                            if p1[2] != p0[2]:
                                a = (p1[1] - p0[1]) / (p1[2] - p0[2])

                                line2 = []
                                lines = [3, 1]
                                for x in range(int(math.ceil(math.fabs(p1[2] - p0[2]))) * mult_factor + 1):
                                    # +1 because have to add the difference for the down boundaries
                                    line2.append(
                                        [int(round(-a * x / mult_factor + p0[1]) + 1), int(round(p0[2] - x / mult_factor))])
                                interpolation = [[[p0[1] + 1, p0[2]]], line2]

                    else:
                        # the lines will never meet so we just go from one point to the other one
                        if p1[2] != p0[2]:
                            a = (p1[1] - p0[1]) / (p1[2] - p0[2])

                            line2 = []
                            lines = [3, 1]
                            for x in range(int(math.ceil(math.fabs(p1[2] - p0[2]))) * mult_factor + 1):
                                # +1 because have to add the difference for the down boundaries
                                line2.append([int(round(-a * x / mult_factor + p0[1]) + 1), int(round(p0[2] - x / mult_factor))])
                            interpolation = [[[p0[1] + 1, p0[2]]], line2]


            elif p1[0] == 3:
                #has same angle
                if p1[1] != p0[1]:
                    a = (p1[2] - p0[2]) / (p1[1] - p0[1])
                    lines = [3]
                    for x in range(int(math.ceil(math.fabs(p1[1] - p0[1]))) * mult_factor + 1):
                        #right border are considered positive
                        interpolation.append([int(p0[1] + x / mult_factor), int(round(a * x / mult_factor + p0[2]))])
                    #todo has not same angle bound

    else:
        print("p0: ", p0, " ind0: ", ind0)
        print("p1: ", p1, " ind1: ", ind1)
        eas = EAS(p0[2], p0[1], math.radians(ind0[1]), p1[2], p1[1], math.radians(ind1[1]), 80)
        interpolation, lines = eas.construct_eas(p0[0])


    if len(interpolation) == 0:
        print("problem with interpolation! curve_completion.py met. find_completion_curve ")
        return None, 0
    else:
        if np.shape(lines)[0] == 1:
            #todo somehow find a way to control every shape of interpolation
            if print_lab:
                print("interpolation parameters for single lines: ")
                print(interpolation)

            if p0[1] == interpolation[0][0] and p0[2] == interpolation[0][1] and p1[1] == interpolation[-1][0] and \
                            p1[2] == interpolation[-1][1]:
                return interpolation, lines

            else:
                print("Problem with the completion curve in single line!!")
                return interpolation, lines
                # return [[0, 0]]
        elif np.shape(lines)[0] == 2:
            #means there's two tables
            if print_lab:
                print("interpolation parameters for multiple lines: ")
                print(interpolation)

            # if p0[1] == interpolation[0][0][0] and p0[2] == interpolation[0][0][1] and p1[1] == interpolation[1][-1][0] and \
            #                 p1[2] == interpolation[1][-1][1]:
            #     return interpolation, lines
            # else:
            #     print("Problem with the completion curve in multiple lines!!")
            return interpolation, lines
        else:
            print("problem with the number of lines -> curve ?")


if __name__ == '__main__':

    # # ========== test horizontal straight line ==========#
    # p0 = np.array([0, 0, 0])
    # p1 = np.array([0, 0, 5])
    # ind0 = [True, 0, 0, 0]
    # ind1 = [True, 0, 0, 0]
    #
    # find_completion_curve(p0, ind0, p1, ind1, True)
    #
    # # ========== test horizontal straight line with diff ==========#
    # # p0 = np.array([[0, 0], [1, 0], [2, 0]])
    # # p1 = np.array([[10, -4], [11, -4], [12, -4]])
    # p0 = np.array([0, 0, 0])
    # p1 = np.array([0, 3, 5])
    # ind0 = [True, 0, 0, 0]
    # ind1 = [True, 0, 0, 0]
    #
    # find_completion_curve(p0, ind0, p1, ind1, True)
    #
    # # ========== test horizontal straight line with angle same level ==========#
    # p0 = np.array([0, 0, 2])
    # p1 = np.array([0, 2, 7])
    # ind0 = [True, 0, 0, [0]]
    # ind1 = [True, 45, 0, [1]]
    #
    # find_completion_curve(p0, ind0, p1, ind1, True)
    #
    # # ========== test horizontal straight line with exit on bound 1 (bottom) -> square ==========#
    # p0 = np.array([0, 0, 0])
    # p1 = np.array([1, 5, 2])
    # ind0 = [True, 0, 0, [0]]
    # ind1 = [True, 0, 0, [0]]
    #
    # find_completion_curve(p0, ind0, p1, ind1, True)
    #
    # # ========== test horizontal straight line with exit on bound 1 (bottom) -> triangle ==========#
    # p0 = np.array([0, 0, 0])
    # p1 = np.array([1, 5, 0])
    # # p0 = np.array([0, 61, 74])
    # # p1 = np.array([1, 88, 74])
    # ind0 = [True, -27, 0, [-0.52]]
    # ind1 = [True, 27, 0, [0.52]]
    #
    # find_completion_curve(p0, ind0, p1, ind1, True)
    #
    # # ========== test horizontal straight line with vertical straight line -> corner ==========#
    # p0 = np.array([0, 0, 0])
    # p1 = np.array([3, 5, 7])
    # ind0 = [True, 0, 0, [0]]
    # # ind0 = [True, 45, 0, [1]]
    # # ind0 = [True, -45, 0, [-1]]
    # ind1 = [True, 90, 0, [math.inf]]
    # # ind1 = [True, 45, 0, [1]]
    # # ind1 = [True, -45, 0, [-1]]
    #
    # find_completion_curve(p0, ind0, p1, ind1, True)

    # # ========== test horizontal triangle ==========#
    # p0 = np.array([0, 0, 0])
    # p1 = np.array([0, 0, 5])
    # ind0 = [True, 45, 0, 0]
    # ind1 = [True, -45, 0, 0]
    # print()
    # print("ind0: ", ind0)
    # print("ind1: ", ind1)
    # print("p0: ", p0)
    # print("p1: ", p1)
    #
    # find_completion_curve(p0, ind0, p1, ind1, True)
    #
    # # ========== test2 horizontal triangle ==========#
    # p0 = np.array([0, 0, 0])
    # p1 = np.array([0, 0, 5])
    # ind0 = [True, 30, 0, 0]
    # ind1 = [True, -65, 0, 0]
    # print()
    # print("ind0: ", ind0)
    # print("ind1: ", ind1)
    # print("p0: ", p0)
    # print("p1: ", p1)
    #
    # find_completion_curve(p0, ind0, p1, ind1, True)

    # ========== test vertical straight line ==========#
    # ind1 = np.array([[0, 0], [0, 1], [0, 2]])
    # ind2 = np.array([[0, 10], [0, 11], [0, 12]])
    # num_lines = 0
    #
    # find_completion_curve(ind1, ind2, True)

    # # ========== test horizontal circle ==========#
    # p0 = np.array([[0, 0], [1, 1], [1, 2]])
    # p1 = np.array([[10, 1], [11, 0]])
    # ind0 = [False, 0, 0, 0]
    # ind1 = [False, -180, 0, 0]
    #
    # find_completion_curve(p0, ind0, p1, ind1, True)

    # ========== test bottom square ==========#
    p0 = np.array([3, 0, 5])
    p1 = np.array([2, 0, 0])
    ind0 = [True, -90, 0, [math.inf]]
    ind1 = [True, -90, 0, [math.inf]]

    find_completion_curve(p0, ind0, p1, ind1, True)

    # ========== test bottom triangle ==========#
    p0 = np.array([3, 11, 20])
    p1 = np.array([2, 11, 10])
    ind0 = [True, 45, 0, [1]]
    ind1 = [True, -45, 0, [-1]]

    find_completion_curve(p0, ind0, p1, ind1, True)

    # ========== test top triangle ==========#
    p0 = np.array([2, 11, 20])
    p1 = np.array([3, 11, 10])
    ind0 = [True, 45, 0, [1]]
    ind1 = [True, -45, 0, [-1]]

    find_completion_curve(p0, ind0, p1, ind1, True)

    # ========== test inner down==========#
    p0 = np.array([3, 0, 0])
    p1 = np.array([2, 0, 5])
    ind0 = [True, 90, 0, [math.inf]]
    ind1 = [True, 90, 0, [math.inf]]

    find_completion_curve(p0, ind0, p1, ind1, True)

    # ========== upper triangle ==========#
    p0 = np.array([2, 0, 0])
    p1 = np.array([3, 0, 5])
    ind0 = [True, 90, 0, [math.inf]]
    ind1 = [True, -45, 0, [-1]]

    find_completion_curve(p0, ind0, p1, ind1, True)

    plt.show()