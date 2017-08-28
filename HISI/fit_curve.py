import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def lin(x, a, b):
    return a * x + b


def quad(x, a, b, c):
    return a * x * x + b * x + c


def find_curve_param(xdata, ydata, use_quadratic, print_lab=False):
    if print_lab:
        print("")

    popt_lin, pcov_lin = curve_fit(lin, xdata, ydata)
    ssr_lin = 0
    ssto_lin = 0

    if use_quadratic:
        popt_quad, pcov_quad = curve_fit(quad, xdata, ydata)
        ssr_quad = 0

    x_ave = np.mean(xdata)
    y_ave = np.mean(ydata)
    for i, x in enumerate(xdata):
        ssr_lin += math.pow(lin(x, popt_lin[0], popt_lin[1]) - y_ave, 2)
        if use_quadratic:
            ssr_quad += math.pow(quad(x, popt_quad[0], popt_quad[1], popt_quad[2]) - y_ave, 2)
        ssto_lin += math.pow(ydata[i] - y_ave, 2)

    if ssto_lin != 0:
        r2_lin = ssr_lin/ssto_lin
        if use_quadratic:
            r2_quad = ssr_quad/ssto_lin
        else:
            r2_quad = 0
    else:
        r2_lin = 0
        r2_quad = 0

    if print_lab:
        print("r2_lin: ", r2_lin, popt_lin, " r2_quad: ", r2_quad, popt_quad)

    if use_quadratic:
        if math.fabs(popt_quad[0]) < 0.05 or r2_lin >= r2_quad:
            #considered linear
            if x_ave == 0:
                if print_lab:
                    print("the curve is considered linear, with an angle of: ", 90)
                return True, 90, 0.0, [0.0, 0.0]
            else:
                angle_lin = math.atan(popt_lin[0]) * 180 / math.pi
                if print_lab:
                    print("the curve is considered linear, with an angle of: ", angle_lin)
                return True, angle_lin, 0.0, popt_lin
        else:
            #considered quadratic
            angle_quad = math.atan(popt_quad[1]) * 180 / math.pi
            radius = math.pow(1 + math.pow(popt_quad[1], 2), 1.5) / np.abs(2 * popt_quad[0])
            if print_lab:
                print("the curve is considered quadratic, with an angle of: ", angle_quad, " and a radius of: ", radius)

            return False, angle_quad, radius, popt_quad

    else:
        if x_ave == 0:
            if print_lab:
                print("the curve is considered linear, with an angle of: ", 90)

            return True, 90, 0.0, [float('Inf')]

        else:
            angle_lin = math.atan(popt_lin[0]) * 180 / math.pi
            if print_lab:
                print("the curve is considered linear, with an angle of: ", angle_lin)

            return True, angle_lin, 0.0, popt_lin


if __name__ == '__main__':

    xdata = np.arange(10)

    # ========== Test case linear function ==========#
    ydata = [0, 1, 2.3, 2.7, 4.3, 4.6, 6.2, 7.1, 7.6, 8.7]

    # xdata = np.array([0,-1,-2,-3])
    # ydata = np.array([0,0,0,0])

    plt.figure()
    is_linear, angle, radius, popt = find_curve_param(xdata, ydata, use_quadratic=False, print_lab=True)
    if is_linear:
        plt.plot(xdata, lin(xdata, *popt), 'r-', label='fit')
    else:
        plt.plot(xdata, quad(xdata, *popt), 'r-', label='fit')

    plt.plot(xdata, ydata, 'b-', label='data')

    # ========== Test case quadratic function ==========#
    plt.figure()
    xdata = np.arange(11)
    ydata = np.array([4, 8, 17, 26, 34, 53, 62, 85, 101, 125, 145])
    ydata -= 4

    is_linear, angle, radius, popt = find_curve_param(xdata, ydata, use_quadratic=True, print_lab=True)
    if is_linear:
        plt.plot(xdata, lin(xdata, *popt), 'r-', label='fit')
    else:
        plt.plot(xdata, quad(xdata, *popt), 'r-', label='fit')

    plt.plot(xdata, ydata, 'b-', label='data')

    # ========== Test case horizontal function ==========#
    plt.figure()
    xdata = np.arange(11)
    ydata = np.zeros(11)

    is_linear, angle, radius, popt = find_curve_param(xdata, ydata, use_quadratic=True, print_lab=True)
    if is_linear:
        plt.plot(xdata, lin(xdata, *popt), 'r-', label='fit')
    else:
        plt.plot(xdata, quad(xdata, *popt), 'r-', label='fit')

    plt.plot(xdata, ydata, 'b-', label='data')

    # ========== Test case vertical function ==========#
    plt.figure()
    xdata = np.zeros(11)
    ydata = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    is_linear, angle, radius, popt = find_curve_param(xdata, ydata, use_quadratic=True, print_lab=True)
    if is_linear:
        plt.plot(xdata, lin(xdata, *popt), 'r-', label='fit')
    else:
        plt.plot(xdata, quad(xdata, *popt), 'r-', label='fit')

    plt.plot(xdata, ydata, 'b-', label='data')

    plt.show()