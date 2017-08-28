import numpy as np
from scipy import signal
from scipy import misc
import math
import matplotlib.pylab as plt
from show_matrix import *
from createFilters import *

# input = np.array([[0.1,0.1,0.4,.4,0.1,0.1],[0.1,0.1,.4,.4,0.1,0.1],[0.1,.92,.92,.92,.92,0.1],[0.1,0.92,0.92,0.92,.92,0.1],[0.1,0.1,.6,.6,0.1,0.1],[0.1,0.1,.6,.6,0.1,0.1]])
input = np.array([[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.]])
print("input")
print(input)

#gaussian filter
def g_f(x, y, a, gamma):
    return a * np.exp(-(x*x + y*y) / (gamma*gamma))

#oriented gaussian filter
def g_f_o(x, y, gamma, m, n):
    return np.exp(-(np.power(x + 0.5, 2) + np.power(y + 0.5, 2)) / (gamma * gamma)) \
           - np.exp(-(np.power(x + 0.5 + m, 2) + np.power(y + 0.5 + n, 2)) / (gamma * gamma))

#m orientation
def m_orien(k):
    return int(math.sin(2 * math.pi * k / 4))

#n orientation
def n_orien(k):
    return int(math.cos(2 * math.pi * k / 4))

a_on = 5
gamma_on = 0.5
a_off = 0.25
gamma_off = 2
size_filter = 3

on_filter = np.array(
    [[g_f(i, j, a_on, gamma_on) for i in range(math.ceil(-size_filter / 2), math.ceil(size_filter / 2))]
     for j in range(math.ceil(-size_filter / 2), math.ceil(size_filter / 2))])
off_filter = np.array(
    [[g_f(i, j, a_off, gamma_off) for i in range(math.ceil(-size_filter / 2), math.ceil(size_filter / 2))]
     for j in range(math.ceil(-size_filter / 2), math.ceil(size_filter / 2))])

excit = signal.convolve2d(input,on_filter, boundary='symm', mode='same')
inhib = signal.convolve2d(input,off_filter, boundary='symm', mode='same')

a = 50
b = 150
z = b * (excit - inhib) / (a + excit + inhib)
print("z")
print(z)

f = np.array([[[g_f_o(i, j, 1.75, m_orien(k), n_orien(k)) for i in range(math.ceil(-size_filter / 2), math.ceil(size_filter / 2))]
     for j in range(math.ceil(-size_filter / 2), math.ceil(size_filter / 2))] for k in range(4)])

print("f")
print(f)
print(np.shape(f))

v = np.array([signal.convolve2d(z,f[i], boundary='symm', mode='same') for i in range(4)])
print("boundary")
print(v)
print(np.shape(v))

max_bound = np.max(v)
v = v / max_bound
v = np.abs(v)

print(v)

plt.figure()
plt.imshow(show_matrix(input, v))


filter0_1, filter0_2 = createFilters(2, 3, 1, 1, 1)
print("filters0")
print(filter0_1)
print(filter0_2)
boundaries2 = np.zeros((4,6,6))
print(np.shape(boundaries2), np.shape(filter0_1))
boundaries2[0,:,:] = signal.convolve2d(input,filter0_1[0,:,:], boundary='symm', mode='same')

plt.figure()
plt.imshow(show_matrix(input, boundaries2))
plt.show()

