"""
Implementation by Michael Stettler

with the kind permission of Hailing Zhou for my master thesis:

Copyright (c) Hailing Zhou

The code is used for research.
Please do not redistribute it and use it for any commercial purposes.

"""
import matplotlib.pylab as plt
from clothoid import *

class EAS():
    def __init__(self,xA, yA, thetaA, xB, yB, thetaB, n):
        self.xA = xA
        self.yA = yA
        self.thetaA = thetaA
        self.xB = xB
        self.yB = yB
        self.thetaB = thetaB
        self.n = n

    def _cal_alpha(self, k, s):
        return 2 * (self.thetaB - self.thetaB - self.n * k * s) / ((self.n - 1) * self.n * s * s)

    def _cal_delta_k(self, i, k, l, clothoid):
        lambda1, lambda2 = clothoid.cal_lamda12(k, l)
        vix, viy = clothoid.cal_vi(i, k, l)

        return -(lambda1 * vix + lambda2 * viy)/2

    def _cal_pi_eas(self, n, k, l, alpha, clothoid):
        #construct euler arc spline (eas)
        pix = 0
        piy = 0
        pix += self.xA
        piy += self.yA

        s = l / self.n

        for i in range(1, n + 1):
            k_i = alpha * s * s * (i - 1) + k
            phi_i = clothoid._cal_feta_i(i, k, l)
            pix += 2 * math.sin(k_i * s / 2) * math.cos(phi_i + self.thetaA) / k_i
            piy += 2 * math.sin(k_i * s / 2) * math.sin(phi_i + self.thetaA) / k_i

        return pix, piy

    def _cal_pi_g1(self, n, k, l, clothoid):
        #construct G1 arc spline
        pix = 0
        piy = 0
        pix += self.xA
        piy += self.yA

        s = l / self.n

        for i in range(1, n + 1):
            delta_theta = clothoid._cal_delta_theta_i(i, k, l)
            phi_i = clothoid._cal_feta_i(i, k, l)
            pix += s * 2 * math.sin(delta_theta / 2) * math.cos(phi_i + self.thetaA) / delta_theta
            piy += s * 2 * math.sin(delta_theta / 2) * math.sin(phi_i + self.thetaA) / delta_theta

        return pix, piy

    def cal_eas(self):
        # step 1: Calculate optimization paremeter using clothoid
        clothoid = Clothoid(self.xA, self.yA, self.thetaA, self.xB, self.yB, self.thetaB, self.n)
        kl = clothoid.clothoid_completion_LM()

        print("Optimization parameters kl = ", kl)
        k = kl[0]
        l = kl[1]

        # step 2: Compute alpha
        alpha = self._cal_alpha(k, l/self.n)

        # step 3: Compute all delta_k
        delta_k = []
        for i in range(self.n):
            delta_k.append(self._cal_delta_k(i, k, l, clothoid))


        # step 4: reconstruct
        eas = True
        if np.shape(np.nonzero(delta_k))[1] > 0:
            eas = False

        x = []
        y = []

        x.append(self.xA)
        y.append(self.yA)

        for i in range(self.n):
            if eas:
                #all delta are equal to 0
                # print("construct EAS")
                p_i_x, p_i_y = self._cal_pi_eas(i, k, l, alpha, clothoid)
            else:
                # print("construct G1")
                p_i_x, p_i_y = self._cal_pi_g1(i, k, l, clothoid)

            x.append(p_i_x)
            y.append(p_i_y)

        x.append(self.xB)
        y.append(self.yB)

        return x, y

    def construct_eas(self, type_bound):
        print("xA", self.xA, " yB", self.yB, " ya", self.xB, " yB", self.yB)

        x, y = self.cal_eas()

        line1 = []
        lines = []

        if type_bound == 0:
            lines.append(3)
        elif type_bound == 1:
            lines.append(2)
        elif type_bound == 2:
            lines.append(0)
        elif type_bound == 3:
            lines.append(1)
        else:
            print("problem with boundary type in euler_arc_spline method: construct_eas()")


        for i in range(self.n):
            line1.append([int(y[i]), int(x[i])])

        interpolation = line1

        # # to show the reconstructed line
        # plt.figure()
        # plt.plot(x, y)
        # plt.show()

        return interpolation, lines

if __name__ == '__main__':
    n = 80

    # # Zhou example
    # xA = 3.90
    # yA = 3.16
    # thetaA = 2.52043
    # xB = 6.43
    # yB = 3.16
    # thetaB = -2.18971

    # quarter circle
    xA = 0.0
    yA = 30.0
    thetaA = 0.0
    xB = 30.0
    yB = 0.0
    thetaB = -3.147/2

    eas = EAS(xA, yA, thetaA, xB, yB, thetaB, n)
    x, y = eas.cal_eas()

    plt.figure()
    plt.plot(x, y)
    plt.show()