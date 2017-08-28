import numpy as np
from sympy.solvers import solve
import math


def _alpha(n, eta, s, thetaA, thetaB):
    return 2 * (thetaB - thetaA - n * eta) / ((n - 1) * n * s * s)


def _phi(n, eta, s, alpha):
    phi = []

    for j in range(n):
        phi.append((np.power(s * (j - 1), 2) * alpha + (2 * j - 1) * eta)/2)

    return phi


def _delta_theta(n, alpha, eta, s):
    d_theta = []

    for i in range(n):
        d_theta.append(alpha * (i - 1) * s * s + eta)
    return d_theta


def _vX(n, delta_theta, thetaA, phi):
    vX = []

    for j in range(n):
        vX.append(2 * np.sin(delta_theta[j] / 2) * np.cos(thetaA + phi[j]) / delta_theta[j])
    return vX


def _vY(n, delta_theta, thetaA, phi):
    vY = []

    for j in range(n):
        vY.append(2 * np.sin(delta_theta[j] / 2) * np.sin(thetaA + phi[j]) / delta_theta[j])
    return vY


def _pB(pA, n, eta, s, phi, thetaA):
    sum0 = 0
    sum1 = 0

    for j in range(n):
        sum0 += 2 * np.sin(eta  / 2) * np.cos(phi[j] + thetaA) / (eta / s)
        sum1 += 2 * np.sin(eta  / 2) * np.sin(phi[j] + thetaA) / (eta / s)

    return [pA[0] + sum0, pA[1] + sum1]


def _p_delta(pA, n, s, delta_, delta_theta, thetaA, phi):
    sum0 = 0
    sum1 = 0

    for j in range(n):
        sum0 += (s + delta_[j]) * 2 * np.sin(delta_theta[j] / 2) * np.cos(thetaA + phi[j]) / delta_theta[j]
        sum1 += (s + delta_[j]) * 2 * np.sin(delta_theta[j] / 2) * np.sin(thetaA + phi[j]) / delta_theta[j]

    return [pA + sum0, pA + sum1]

def _lambda(vX, vY, pA, pB, s):

    v_X = 0
    v_Y = 0
    vX2 = 0
    vY2 = 0
    vXY = 0

    for i in range(len(vX)):
        v_X += vX[i]
        v_Y += vY[i]
        vX2 += vX[i] * vX[i]
        vY2 += vY[i] * vY[i]
        vXY += vX[i] * vY[i]

    denom = vX2 * vY2 - vXY * vXY

    return [2 * vY2 * (pA[0] - pB[0] + s * v_X - 2 * vXY * (pA[1] - pB[1] + s * v_Y)) / denom,
            2 * vX2 * (pA[1] - pB[1] + s * v_Y - 2 * vXY * (pA[0] - pB[0] + s * v_X)) / denom]

def _delta(n, eta, s, thetaA, thetaB, p0):
    delta = []

    alpha = _alpha(n, eta, s, thetaA, thetaB)
    delta_theta = _delta_theta(n, alpha, eta, s)
    phi = _phi(n, eta, s, alpha)
    vX = _vX(n, delta_theta, thetaA, phi)
    vY = _vY(n, delta_theta, thetaA, phi)
    pA = p0
    pB = _pB(pA, n, eta, s, phi, thetaA)
    lambda_ = _lambda(vX, vY, pA, pB, s)

    for i in range(n):
        delta.append(- (lambda_[0] * vX[i] + lambda_[1] * vY[i]) / 2)
    return delta


def init_s(p0, p1, n):
    return 1.5 * np.sqrt((np.power(p0[0] - p1[0], 2) + np.power(p0[0] - p1[0], 2))) / n


def find_opti_s_n(s, eta, f_delta, p1):
    print("bouh")
    n = np.shape(f_delta)[0]
    nu = 2
    mu = 10e-3
    epsi1 = 10e-10
    epsi2 = 10e-10
    stop = False
    k_max = 100
    k = 0
    J = np.zeros((n, 2))
    A = np.transpose(J) @ J + mu * np.diag(np.transpose(J) @ J)
    print("A ", np.shape(A))
    print(A)
    G = - np.dot(np.transpose(J), f_delta)
    print("G ", np.shape(G))
    print(G)

    while k < k_max and not stop:
        k += 1
        d = 0
        while d <= 0 and not stop:
            B = (A + mu * np.diag(np.transpose(J) @ J)) @ np.transpose(np.array([1,1]))
            print("B ", np.shape(B), B)
            delta_s = solve(B[0], G[0])
            delta_eta = solve(B[1], G[1])
            if math.pow(delta_s - delta_eta, 2) <= epsi1 * math.pow(s - eta, 2):
                stop = True
                print("if is true ?")
            else:
                h0 = min(1, p1[0] - s)
                h1 = min(1, p1[1] - eta)
                s_new = s + h0 * delta_s
                eta_new = eta + h1 * delta_eta
                # f_new =
            print(a, b)
            d += 1

    return 0, 0

def construct_EAS(p0, theta0, p1, theta1):

    n = 10
    s0 = init_s(p0, p1, n)
    eta0 = 1
    f_delta = _delta(n, eta0, s0, theta0, theta1, p0)
    print(f_delta)

    # s, eta = find_opti_s_n(s0, eta0, f_delta, p1)

if __name__ == '__main__':

    p0 = [0, 0]
    theta0 = 45
    p1 = [5, 0]
    theta1 = -45

    construct_EAS(p0, theta0, p1, theta1)
