import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from show_matrix import *


class TBE(object):

    def __init__(self, p0, k0, p1, k1):
        #tangeant bundle elastica parameters
        self.p0 = p0
        self.k = k0
        self.p1 = p1
        self.k1 = k1

        self.delta = 0.01
        self.h_bar = 1.0 #proportionalité constant -> need to read a previous paper to understand

    def _func(self, params, num_points):
        g = np.zeros(num_points)
        g[0] = params[2]
        k = np.zeros(num_points)
        k[0] = self.k
        s = np.zeros(num_points)
        s[0] = 0.0
        x = np.zeros(num_points)
        x[0] = self.p0[0]
        y = np.zeros(num_points)
        y[0] = self.p0[1]
        theta = np.zeros(num_points)
        theta[0] = self.p0[2]

        n = 0
        while n < num_points - 1:
            s[n + 1] = s[n] + self.delta

            if k[n] == 0:
                g[n + 1] = g[n]
            else:
                g[n + 1] = g[n] + max(
                    min(self.delta * ((math.pow(k[n], 2) - 2 * math.pow(self.h_bar, 2) * math.pow(k[n], 4) -
                                       3 * math.pow(self.h_bar, 4) * math.pow(k[n], 6) +
                                       math.pow(self.h_bar * g[n], 2) +
                                       7 * math.pow(self.h_bar, 4) * math.pow(k[n] * g[n], 2)) /
                                      (2 * math.pow(self.h_bar, 2) * k[n] * (1. + math.pow(self.h_bar * k[n], 2)))
                                      -
                                      # (params[0] * math.sin(self.p0[2] + params[1]) * math.pow(
                                      (params[0] * math.sin(theta[n] + params[1]) * math.pow(
                                          1 + math.pow(self.h_bar * k[n], 2), 3)) /
                                      (2 * math.pow(self.h_bar, 2) * k[n])), 10), -10)

            # print("g[n + 1] ", g[n + 1])
            k[n + 1] = k[n] + self.delta * g[n]
            # print("k[n + 1] ", k[n + 1])
            theta[n + 1] = theta[n] + self.delta * k[n]
            y[n + 1] = y[n] + self.delta * math.sin(theta[n])
            x[n + 1] = x[n] + self.delta * math.cos(theta[n])

            n += 1

        return x, y, theta, k

    def _g(self, params):

        num_points = max(int(params[3] / self.delta), 1)

        x, y, theta, k = self._func(params, num_points)

        # print("last point : (", x[-1], ";", y[-1], ";", theta[-1], ";", k[-1], ")")
        # print("final point : (", self.p1[0], ";", self.p1[1], ";", self.p1[2], ";", self.k1, ")")
        err = np.power(self.p1[0] - x[-1], 2) + np.power(self.p1[1] - y[-1], 2) + np.power(self.p1[2] - theta[-1], 2) +\
              np.power(self.k1 - k[-1], 2)
        # print("error g :", err)

        return err

    def _construct_array(self, params, print_lab=False, name_graph=""):

        num_points = max(int(params[3] / self.delta), 1) + int(1/self.delta)

        x, y, theta, k = self._func(params, num_points)

        new_boundaries = []
        size_array = int(num_points * self.delta)
        for i in range(1, size_array):
            new_boundaries.append([int(x[i / self.delta]), int(y[i / self.delta])])

        if print_lab:
            print("last point : (", x[-1], ";", y[-1], ";", theta[-1], ";", k[-1], ")")
            print("final point : (", self.p1[0], ";", self.p1[1], ";", self.p1[2], ";", self.k1, ")")
            err = np.power(self.p1[0] - x[-1], 2) + np.power(self.p1[1] - y[-1], 2) + np.power(self.p1[2] - theta[-1],
                                                                                               2) + \
                  np.power(self.k1 - k[-1], 2)
            print("error g :", err)

            plt.figure()
            plt.plot(x, y)
            print("name graph : ", name_graph)
            plt.title(name_graph)

            print("new_boundaries")
            print(new_boundaries)
            print("size_array ", size_array)

            img = np.zeros((size_array, size_array))
            for i in range(0, size_array - 1):
                img[new_boundaries[i][0], new_boundaries[i][1]] = 1

            print(img)

        return new_boundaries

    def _set_l(self, p0, p1):
        l = np.sqrt(np.power(p0[0] - p1[0], 2) + np.power(p0[0] - p1[0], 2))
        return l

    def construct_TBE(self, print_lab=False, name_graph=""):


        # alpha = 0.01 #learning rate
        # for i in range(0, 1):
        #     xl, yl, thetal, kl = self._g(p0, k0)
        #
        #     err = np.power(np.linalg.norm([[p1[0], p1[1], p1[2], k1], [xl[-1], yl[-1], thetal[-1], kl[-1]]]), 2)
        #
        #     if i == 100 - 1:
        #         print(err)
        #     # ====== update param here ======
        #     # grad = err ??????
        #     # self.c = self.c - (alpha * grad)
        #     # self.phi = self.phi - (alpha * grad)
        #     # self.g0 = self.g0 - (alpha * grad)
        #     # self.l = self.l - (alpha * grad)

        c0 = .5
        phi0 = 0
        g0 = 0
        l0 = self._set_l(p0, p1)
        # res = minimize(rosen, [p0, k0], method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
        # res = minimize(self._g, [c0, phi0, g0, l0], method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
        # res = minimize(self._g, [c0, phi0, g0, l0], method='BFGS', options={'tol': 1e-8, 'disp': True, 'maxiter':200})

        res = minimize(self._g, [c0, phi0, g0, l0], method='powell', options={'gtol': 1e-8, 'disp': True, 'maxiter':100})
        self._construct_array([res.x[0], res.x[1], res.x[2], res.x[3]], print_lab, name_graph)

        # res = [0.21801552, 1.83157083, -0.24597403, 8.68548678]
        # self._construct_array([res[0], res[1], res[2], res[3]], print_lab, name_graph)


if __name__ == '__main__':
    # # *********** A ************
    #
    # p0 = [-0.5, 0.0, 40 * math.pi / 180]
    # k0 = -.57
    # p1 = [0.5, 0.0, - 50 * math.pi / 180]
    # k1 = -.57
    #
    # tbe = TBE(p0, k0, p1, k1)
    # tbe.construct_TBE(True, "Fig 8 A")
    #
    # # *********** C ************
    #
    # p0 = [-0.5, 0.0, 30 * math.pi / 180]
    # k0 = -.7
    # p1 = [0.5, 0.0, - 50 * math.pi / 180]
    # k1 = -2.3
    #
    # tbe = TBE(p0, k0, p1, k1)
    # tbe.construct_TBE(True, "Fig 8 C")
    #
    # # *********** D ************
    #
    # p0 = [-0.5, 0.0, 20 * math.pi / 180]
    # k0 = -.7
    # p1 = [0.5, 0.0, - 30 * math.pi / 180]
    # k1 = -2
    #
    # tbe = TBE(p0, k0, p1, k1)
    # tbe.construct_TBE(True, "Fig 8 D")

    # ********** Michael's test 1 *********

    # p0 = [0.0, 0.0, 45 * math.pi / 180]
    # k0 = -0.008
    # p1 = [5, 0, -45 * math.pi / 180]
    # k1 = -0.008
    #
    # tbe = TBE(p0, k0, p1, k1)
    # tbe.construct_TBE(True, "Michael's test 1")
    #
    # # ********** Michael's test 2 *********
    #
    # p0 = [0.0, 0.0, 90 * math.pi / 180]
    # k0 = -.25
    # p1 = [2.0, 0.0, -90 * math.pi / 180]
    # k1 = -.25
    #
    # tbe = TBE(p0, k0, p1, k1)
    # tbe.construct_TBE(True, "Michael's test 2")
    #
    # ********** Michael's test 3 *********

    p0 = [0.0, 0.0, 0 * math.pi / 180]
    k0 = -0.0
    p1 = [4.0, 0.0, -180 * math.pi / 180]
    k1 = -0.0

    tbe = TBE(p0, k0, p1, k1)
    tbe.construct_TBE(True, "Michael's test 3")

    # ********** Michael's test 4 *********

    p0 = [7.0, -5.0, 0 * math.pi / 180]
    k0 = -0.009
    p1 = [0.0, 0.0, 0 * math.pi / 180]
    k1 = 0.009

    tbe = TBE(p0, k0, p1, k1)
    tbe.construct_TBE(True, "Michael's test 4")

    plt.show()


