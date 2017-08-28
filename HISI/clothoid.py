"""
Implementation by Michael Stettler

with the kind permission of Hailing Zhou for my master thesis:

Copyright (c) Hailing Zhou

The code is used for research.
Please do not redistribute it and use it for any commercial purposes.

"""

from scipy.optimize import root
import numpy as np
import math

class Clothoid():
    def __init__(self,xA, yA, thetaA, xB, yB, thetaB, n):
        self.xA = xA
        self.yA = yA
        self.thetaA = thetaA
        self.xB = xB
        self.yB = yB
        self.thetaB = thetaB
        self.n = n


    #Calculate Theta_i
    def _cal_theta_i(self, i, k, l):
        return ((i/self.n)-pow((i/self.n), 2))*k*l + self.thetaA + ((self.thetaB-self.thetaA)*pow((i/self.n), 2))


    #Calculate DeltaTheta_i
    def _cal_delta_theta_i(self, i, k, l):
        theta_i = self._cal_theta_i(i, k, l)
        theta_i_1 = self._cal_theta_i(i - 1, k, l)

        return theta_i - theta_i_1

    #Calculate a_i
    def _cal_ai(self, i, k, l):
        dti = self._cal_delta_theta_i(i, k, l)
        return (2.0*math.sin(dti/2.0)) / dti

    #Calculate feta_i
    def _cal_feta_i(self, i, k, l):
        thetai = self._cal_theta_i(i, k, l)
        thetai_1 = self._cal_theta_i(i - 1, k, l)
        return ((thetai + thetai_1) / 2.0) - self.thetaA


    #Calculate V_i
    def cal_vi(self, i, k, l):
        ai = self._cal_ai(i, k, l)
        f_eta_i = self._cal_feta_i(i, k, l)
        vix = (ai * math.cos(self.thetaA) * math.cos(f_eta_i)) - (ai * math.sin(self.thetaA) * math.sin(f_eta_i))
        viy = (ai * math.cos(self.thetaA) * math.sin(f_eta_i)) + (ai * math.sin(self.thetaA) * math.cos(f_eta_i))

        return vix, viy

    #Calculate the term: (L/N)*sum(a_i*M(feta_i)*Ta) about the x and ycomponent
    def _cal_sum_sAMT(self, k, l):
        tsx = 0
        tsy = 0

        for i in range(1, self.n + 1):
            vix, viy = self.cal_vi(i, k, l)
            tsx += vix
            tsy += viy

        s_bar = l / self.n
        tsx = (s_bar * tsx)
        tsy = (s_bar * tsy)

        return tsx, tsy

    #Calculate P_L
    def _cal_PL(self, k, l):
        sumx, sumy = self._cal_sum_sAMT(k, l)
        dx = self.xB - self.xA
        dy = self.yB - self.yA

        plx = dx - sumx
        ply = dy - sumy

        return plx, ply

    #Calculate P_L^bar
    def _cal_PL_bar(self, k, l):
        plx, ply = self._cal_PL(k, l)

        pl_bx = -ply
        pl_by = plx

        return pl_bx, pl_by

    #Calculate P
    def _cal_p(self, k, l):
        plx, ply = self._cal_PL(k, l)
        return (plx * plx) + (ply * ply)

    #Calculate the c_i and b_i
    def _cal_FBC(self, i, k, l):
        plx, ply = self._cal_PL(k, l)
        pl_bx, pl_by = self._cal_PL_bar(k, l)
        vix, viy = self.cal_vi(i, k, l)

        fbi = (plx * vix) + (ply * viy)
        fci = (pl_bx * vix) + (pl_by * viy)

        return fbi, fci


    #Calculate the sum(c_i^2), sum(b_i^2), and sum(b_i*c_i)
    def _cal_F_sum_bc(self, k, l):
        fsumb2 = 0
        fsumc2 = 0
        fsumbc = 0

        for i in range(1, self.n + 1):
            bi, ci = self._cal_FBC(i, k, l)
            fsumb2 += (bi * bi)
            fsumc2 += (ci * ci)
            fsumbc += (bi * ci)

        return fsumb2, fsumc2, fsumbc

    #Calculate the denominator of the lamda1 and lamda2
    def _cal_denominator(self, k, l):
        fsumb2, fsumc2, fsumbc = self._cal_F_sum_bc(k, l)

        return (fsumb2 * fsumc2) - (fsumbc * fsumbc)

    #Calculate lamda1 and lamda2
    def cal_lamda12(self, k, l):
        p = self._cal_p(k, l)
        fsumb2, fsumc2, fsumbc = self._cal_F_sum_bc(k, l)
        deno = self._cal_denominator(k, l)

        lamda1 = (-2.0 * p * fsumc2) / deno
        lamda2 = (2.0 * p * fsumbc) / deno

        return lamda1, lamda2


    def _define_functions_of_epsi(self, kl):

        epsi = np.zeros(self.n)
        lambda1, lambda2 = self.cal_lamda12(kl[0], kl[1])

        for i in range(self.n):
            fbi, fci = self._cal_FBC(i, kl[0], kl[1])
            epsi[i] = (-0.5) * ((lambda1 * fbi) + (lambda2 * fci))

        return epsi

    def _cal_sum_of_squared_epsi(self, k, l):
        lamda1, lamda2 = self.cal_lamda12(k, l)

        sum = 0
        for i in range(self.n):
            fbi, fci = self._cal_FBC(i, k, l)
            epsi = -0.5 * ((lamda1 * fbi) + (lamda2 * fci))
            sum += (epsi * epsi)

        return sum


    #######################################################################################
    #############################   Jacobian eosi  #######################################
    #######################################################################################

    def _cal_deriv_theta_i(self, i, k, l):
        de_thetai_k = (((i/self.n)-((i*i)/(self.n*self.n)))*l)
        de_thetai_l = (((i/self.n)-((i*i)/(self.n*self.n)))*k)

        return de_thetai_k, de_thetai_l

    #Calculate DeltaTheta_i
    def _cal_deriv_delta_theta_i(self, i, k, l):
        de_thetai_k, de_thetai_l = self._cal_deriv_theta_i(i, k, l)
        de_thetai_1_k, de_thetai_1_l = self._cal_deriv_theta_i(i - 1, k, l);

        de_dthetai_k = de_thetai_k - de_thetai_1_k
        de_dthetai_l = de_thetai_l - de_thetai_1_l

        return de_dthetai_k, de_dthetai_l

    #Calculate a_i
    def _cal_deriv_ai(self, i, k, l):
        dti = self._cal_delta_theta_i(i, k, l)
        de_dthetai_k, de_dthetai_l = self._cal_deriv_delta_theta_i(i, k, l)

        de_ai_k = ((math.cos(dti / 2.0) / dti) - (2.0 * math.sin(dti / 2.0) / (dti * dti))) * de_dthetai_k
        de_ai_l = ((math.cos(dti / 2.0) / dti) - (2.0 * math.sin(dti / 2.0) / (dti * dti))) * de_dthetai_l

        return de_ai_k, de_ai_l

    #Calculate feta_i
    def _cal_deriv_f_eta_i(self, i, k, l):
        de_thetai_k, de_thetai_l = self._cal_deriv_theta_i(i, k, l)
        de_thetai_1_k, de_thetai_1_l = self._cal_deriv_theta_i(i - 1, k, l)

        de_fetai_k = 0.5 * (de_thetai_k + de_thetai_1_k)
        de_fetai_l = 0.5 * (de_thetai_l + de_thetai_1_l)

        return de_fetai_k, de_fetai_l

    #Calculate V_i
    def _cal_deriv_vi(self, i, k, l):
        ai = self._cal_ai(i, k, l)
        fetai = self._cal_feta_i(i, k, l)

        de_ai_k, de_ai_l = self._cal_deriv_ai(i, k, l)
        de_fetai_k, de_fetai_l = self._cal_deriv_f_eta_i(i, k, l)

        de_vix_k = (-ai * de_fetai_k * ((math.cos(self.thetaA) * math.sin(fetai)) + (math.sin(self.thetaA) * math.cos(fetai)))) + (de_ai_k * ((math.cos(self.thetaA) * math.cos(fetai)) - (math.sin(self.thetaA) * math.sin(fetai))))
        de_viy_k = (ai * de_fetai_k * ((math.cos(self.thetaA) * math.cos(fetai)) - (math.sin(self.thetaA) * math.sin(fetai)))) + (de_ai_k * ((math.cos(self.thetaA) * math.sin(fetai)) + (math.sin(self.thetaA) * math.cos(fetai))))

        de_vix_l = (-ai * de_fetai_l * ((math.cos(self.thetaA) * math.sin(fetai)) + (math.sin(self.thetaA) * math.cos(fetai)))) + (de_ai_l * ((math.cos(self.thetaA) * math.cos(fetai)) - (math.sin(self.thetaA) * math.sin(fetai))))
        de_viy_l = (ai * de_fetai_l * ((math.cos(self.thetaA) * math.cos(fetai)) - (math.sin(self.thetaA) * math.sin(fetai)))) + (de_ai_l * ((math.cos(self.thetaA) * math.sin(fetai)) + (math.sin(self.thetaA) * math.cos(fetai))))

        return de_vix_k, de_viy_k, de_vix_l, de_viy_l

    #Calculate the term: (L/N)*sum(a_i*M(feta_i)*Ta) about the x and ycomponent
    def _cal_deriv_sum_sAMT(self, k, l):
        de_sx_k = 0
        de_sy_k = 0
        de_sx_l = 0
        de_sy_l = 0
        sx = 0
        sy = 0

        for i in range(self.n):
            vix, viy = self.cal_vi(i, k, l)
            sx += vix
            sy += viy

            de_vix_k, de_viy_k, de_vix_l, de_viy_l = self._cal_deriv_vi(i, k, l)
            de_sx_k += de_vix_k
            de_sy_k += de_viy_k
            de_sx_l += de_vix_l
            de_sy_l += de_viy_l

        de_sx_k = ((l / self.n)*de_sx_k)
        de_sy_k = ((l / self.n)*de_sy_k)
        de_sx_l = ((l / self.n)*de_sx_l)
        de_sy_l = ((l / self.n)*de_sy_l)

        #Since the s_bar = L / N, which include the variable of L.So Deriv(s_bar)_L = 1 / L, Deriv(s_bar)_k0 = 0;
        de_sx_l = de_sx_l + ((1 / self.n)*sx)
        de_sy_l = de_sy_l + ((1 / self.n)*sy)

        return de_sx_k, de_sy_k, de_sx_l, de_sy_l

    #Calculate P_L
    def _cal_deriv_PL(self, k, l):
        de_sumx_k, de_sumy_k, de_sumx_l, de_sumy_l = self._cal_deriv_sum_sAMT(k, l)

        de_plx_k = -de_sumx_k
        de_ply_k = -de_sumy_k
        de_plx_l = -de_sumx_l
        de_ply_l = -de_sumy_l

        return de_plx_k, de_ply_k, de_plx_l, de_ply_l

    #Calculate P_L^bar
    def _cal_deriv_PL_bar(self, k, l):
        de_sumx_k, de_sumy_k, de_sumx_l, de_sumy_l = self._cal_deriv_sum_sAMT(k, l)

        de_pl_bx_k = de_sumy_k
        de_pl_by_k = -de_sumx_k
        de_pl_bx_l = de_sumy_l
        de_pl_by_l = -de_sumx_l

        return de_pl_bx_k, de_pl_by_k, de_pl_bx_l, de_pl_by_l

    #Calculate P
    def _cal_deriv_P(self, k, l):
        px, py = cal_PL(k, l)
        de_px_k, de_py_k, de_px_l, de_py_l = self._cal_deriv_PL(k, l)

        de_p_k = (2.0 * px * de_px_k) + (2.0 * py * de_py_k)
        de_p_l = (2.0 * px * de_px_l) + (2.0 * py * de_py_l)

        return de_p_k, de_p_l

    #Calculate the c_i and b_i
    def _cal_deriv_FBC(self, i, k, l):
        px, py = self._cal_PL(k, l)
        p_bx, p_by = self._cal_PL_bar(k, l)
        vix, viy = self.cal_vi(i, k, l)

        de_px_k, de_py_k, de_px_l, de_py_l = self._cal_deriv_PL(k, l)
        de_p_bx_k, de_p_by_k, de_p_bx_l, de_p_by_l = self._cal_deriv_PL_bar(k, l)
        de_vix_k, de_viy_k, de_vix_l, de_viy_l = self._cal_deriv_vi(i, k, l)

        de_fbi_k = ((de_px_k * vix) + (px * de_vix_k)) + ((de_py_k * viy) + (py * de_viy_k))
        de_fbi_l = ((de_px_l * vix) + (px * de_vix_l)) + ((de_py_l * viy) + (py * de_viy_l))

        de_fci_k = ((de_p_bx_k * vix) + (p_bx * de_vix_k)) + ((de_p_by_k * viy) + (p_by * de_viy_k))
        de_fci_l = ((de_p_bx_l * vix) + (p_bx * de_vix_l)) + ((de_p_by_l * viy) + (p_by * de_viy_l))

        return de_fbi_k, de_fbi_l, de_fci_k, de_fci_l

    #Calculate the sum(c_i^2), sum(b_i^2), and sum(b_i*c_i)
    def _cal_deriv_F_sum_bc(self, k, l):
        de_fsumb2_k = 0
        de_fsumb2_l = 0
        de_fsumc2_k = 0
        de_fsumc2_l = 0
        de_fsumbc_k = 0
        de_fsumbc_l = 0

        for i in range(self.n):
            bi, ci = self._cal_FBC(i, k, l)
            de_bi_k, de_bi_l, de_ci_k, de_ci_l = self._cal_deriv_FBC(i, k, l)
            de_fsumb2_k += (2.0 * bi * de_bi_k)
            de_fsumb2_l += (2.0 * bi * de_bi_l)

            de_fsumc2_k += (2.0 * ci * de_ci_k)
            de_fsumc2_l += (2.0 * ci * de_ci_l)

            de_fsumbc_k += (de_bi_k * ci) + (bi * de_ci_k)
            de_fsumbc_l += (de_bi_l * ci) + (bi * de_ci_l)

        return de_fsumb2_k, de_fsumb2_l, de_fsumc2_k, de_fsumc2_l, de_fsumbc_k, de_fsumbc_l

    #Calculate the denominator of the lamda1 and lamda2
    def _cal_deriv_denominator(self, k, l):
        fsumb2, fsumc2, fsumbc = self._cal_F_sum_bc(k, l)

        de_fsumb2_k, de_fsumb2_l, de_fsumc2_k, de_fsumc2_l, de_fsumbc_k, de_fsumbc_l = self._cal_deriv_F_sum_bc(k, l)

        de_deno_k = ((de_fsumb2_k * fsumc2) + (fsumb2 * de_fsumc2_k)) - (2.0 * fsumbc * de_fsumbc_k);
        de_deno_l = ((de_fsumb2_l * fsumc2) + (fsumb2 * de_fsumc2_l)) - (2.0 * fsumbc * de_fsumbc_l);

        return de_deno_k, de_deno_l

    #Calculate lamda1 and lamda2
    def _cal_deriv_lamda12(self, k, l):
        p = self._cal_p(k, l)
        deno = self._cal_denominator(k, l)
        fsumb2, fsumc2, fsumbc = self._cal_F_sum_bc(k, l)

        de_p_k, de_p_l = self._cal_deriv_P(k, l)
        de_deno_k, de_deno_l = self._cal_deriv_denominator(k, l)
        de_fsumb2_k, de_fsumb2_l, de_fsumc2_k, de_fsumc2_l, de_fsumbc_k, de_fsumbc_l = self._cal_deriv_F_sum_bc(k, l)

        de_lamda1_k = (((-2.0 * p * de_fsumc2_k) + (-2.0 * fsumc2 * de_p_k)) / deno) + ((2.0 * p * fsumc2 * de_deno_k) / (deno * deno));
        de_lamda1_l = (((-2.0 * p * de_fsumc2_l) + (-2.0 * fsumc2 * de_p_l)) / deno) + ((2.0 * p * fsumc2 * de_deno_l) / (deno * deno));

        de_lamda2_k = (((2.0 * p * de_fsumbc_k) + (2.0 * fsumbc * de_p_k)) / deno) + ((-2.0 * p * fsumbc * de_deno_k) / (deno * deno));
        de_lamda2_l = (((2.0 * p * de_fsumbc_l) + (2.0 * fsumbc * de_p_l)) / deno) + ((-2.0 * p * fsumbc * de_deno_l) / (deno * deno));

        return de_lamda1_k, de_lamda1_l, de_lamda2_k, de_lamda2_l

    def _define_jacob_of_epsi(self, kl):
        epsi = self._define_functions_of_epsi(kl)

        jac = np.zeros((self.n, 2))
        lamda1, lamda2 = self.cal_lamda12(kl[0], kl[1])
        de_lamda1_k, de_lamda1_l, de_lamda2_k, de_lamda2_l = self._cal_deriv_lamda12(kl[0], kl[1])

        for i in range(self.n):
            fbi, fci = self._cal_FBC(i, kl[0], kl[1])
            de_fbi_k, de_fbi_l, de_fci_k, de_fci_l = self._cal_deriv_FBC(i, kl[0], kl[0])

            jac[i][0] = (-0.5) * (de_lamda1_k * fbi + lamda1 * de_fbi_k + de_lamda2_k * fci + lamda2 * de_fci_k) #the derivation about k0
            jac[i][1] = (-0.5) * (de_lamda1_l * fbi + lamda1 * de_fbi_l + de_lamda2_l * fci + lamda2 * de_fci_l) #the derivation about l

        return jac


    def clothoid_completion_LM(self):

        dist = math.sqrt(math.pow(self.xA - self.xB,2) + math.pow(self.yA - self.yB,2))
        # res = [-1.38058, dist*1.5]
        res = [0, dist*1.5]

        res = root(self._define_functions_of_epsi, res, jac=None, method='lm', options=dict(xtol=0.0000000001))

        return res.x

        # alpha = np.zeros(n)
        # for i in range(n):
        #     alpha[i] = cal_ai(i, res.x[0], res.x[1])

        # epsi = define_functions_of_epsi(x)
        # epsi_opti = define_functions_of_epsi(res.x)

        # plt.plot(np.linspace(xA, xB, n), epsi + yA)
        # plt.plot(np.linspace(xA, xB, n), epsi_opti + yA, 'r')
        # plt.show()




        # print("Iteration number:", int(rep.terminationtype))
        # print("Solution of (k0,l):", x.tostring(10).c_str())

        ##############################################################################
        #For the Jacobian - free LM code:
        # x = [-0.00458603, 379.5]
        # epsg = 0.0000000001
        # epsf = 0
        # epsx = 0
        # maxits = 0
        # state
        # rep
        #
        # minlmcreatev(2, N, x, 0.0001, state)
        # minlmsetcond(state, epsg, epsf, epsx, maxits)
        # minlmoptimize(state, DefineFunctionsOfepsi)
        # minlmresults(state, x, rep)
        #
        # printf("%d\n", int(rep.terminationtype))
        # printf("%s\n\n", x.tostring(10).c_str())

        ##############################################################################
        #For test:

        # k0_est = x_org[0]
        # l_est = x_org[1]
        # sum = cal_sum_of_squared_epsi(k0_est, l_est)
        # print("Sum of squared epsi based on estimated k0 and l:", sum)
        #
        # k0 = x[0], l = x[1]
        # print("Solution of k0 and l:", k0, l)
        # sum = cal_sum_of_squared_epsi(k0, l)
        # print("Sum of squared epsi based on optimization:", sum)
        #
        # print("Solution of k0 and l:", k0, l)


if __name__ == '__main__':
    n = 40

    xA = 3.90
    yA = 3.16
    thetaA = 2.52043
    xB = 6.43
    yB = 3.16
    thetaB = -2.18971

    kl = Clothoid(xA, yA, thetaA, xB, yB, thetaB, n).clothoid_completion_LM()

    print("res:")
    print(kl)  # expected : [-0.62643836  8.39422557]

