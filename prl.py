#!/usr/bin/env python
# -*- coding: utf-8 -*-
### Laser Module
# Define some variables:
from __future__ import division

from math import *
import numpy as np
# from scipy import special
from numpy.lib.scimath import *
import cmath
import scipy.special

beam = 100
name = "Radically Polorized Laser Beam"
E0 =  1
c = 3E8
k0 = 2 * pi / (800 * 1E-9)
a = 20 / k0
w0 = c * k0

# e_hat = 1.0
e_hat = 32.293351126
# e_hat = 331.851851852
# e_hat = 182.844140688
### Er ###
# A0 = E0 * k0**3
A0 = e_hat
### Ez ###
# B0 = E0 * k0**3
B0 = e_hat
### Btheta ###
# C0 = E0 * k0**3/c
C0 = e_hat


z0 = 20.0
phi_0 = pi/4.0
s = 70


# Modified R Tilde
# z0 is a, whhich is confocal parameter
# def R_tilde(t, x, y, z):
#     return np.sqrt( x**2+y**2+z**2-z0**2 + 2*z*z0 * 1j )

def R_tilde(t, x, y, z):
    return np.sqrt( -1.0*(x**2+y**2)-z**2+z0**2 - 2*z*z0 * 1j )* 1j

def R_tilde_2(t, x, y, z):
    return np.sqrt( -1.0*(x**2+y**2)-z**2+z0**2 - 2*z*z0 * 1j )* 1j

# def R_tilde_real(t, x, y, z):
#     return R_tilde(t, x, y, z)

# Bessel Function
# Original Version for j0_real
# def j0(t, x, y, z):
# 	return special.jn(0, R_tilde(t, x, y, z) )
# def j1(t, x, y, z):
# 	return special.jn(1, R_tilde(t, x, y, z) )
# def j2(t, x, y, z):
# 	return special.jn(2, R_tilde(t, x, y, z) )


# sin, cos function part
# def theta(t, x, y, z):
#     return np.arctan( np.sqrt( (x**2+y**2) )/ (z+z0*1j)  )
def theta(t, x, y, z):
    return np.angle(R_tilde(t, x, y, z))

def sin_theta(t, x, y, z):
    return np.sin( theta(t, x, y, z) )


def cos_theta(t, x, y, z):
	return np.cos(theta(t, x, y, z))

def sin_2_theta(t, x, y, z):
	return np.sin(2 * theta(t, x, y, z))

def cos_2_theta(t, x, y, z):
	return np.cos(2 * theta(t, x, y, z))

# P_2 function
# def p2_cos_theta(t, x, y, z):
# 	return 1.0/4 * ( 1 + 3 * cos_2_theta(t, x, y, z) )

# Time term part
# def time(t, x, y, z):
# 	return np.exp( (t + phi_0) * 1j )

def G_0_neg(t, x, y, z):
    R = R_tilde(t, x, y, z)
    e_f = np.exp(1j*phi_0)
    return e_f * ( 1 - 1j * (t - R+1j*z0)/s )**(-s-1) - \
            e_f * ( 1 - 1j * (t + R+1j*z0)/s )**(-s-1)

def G_1_neg(t, x, y, z):
    R = R_tilde(t, x, y, z)
    e_f = np.exp(1j*phi_0)
    return e_f * (-s-1) * ( 1 - 1j * (t - R+1j*z0)/s )**(-s-2) * (-1j/s) -\
    e_f * (-s-1) * ( 1 - 1j * (t + R+1j*z0)/s )**(-s-2) * (-1j/s)

def G_1_pos(t, x, y, z):
    R = R_tilde(t, x, y, z)
    e_f = np.exp(1j*phi_0)
    return e_f * (-s-1) * ( 1 - 1j * (t - R+1j*z0)/s )**(-s-2) * (-1j/s) + \
            e_f * (-s-1) * ( 1 - 1j * (t + R+1j*z0)/s )**(-s-2) * (-1j/s)

def G_2_neg(t, x, y, z):
    R = R_tilde(t, x, y, z)
    e_f = np.exp(1j*phi_0)
    return e_f * (s+1)*(s+2) * ( 1 - 1j * (t - R+1j*z0)/s )**(-s-3) * (1j/s)**2 - \
            e_f * (s+1)*(s+2) * ( 1 - 1j * (t + R+1j*z0)/s )**(-s-3) * (1j/s)**2

def G_2_pos(t, x, y, z):
    R = R_tilde(t, x, y, z)
    e_f = np.exp(1j*phi_0)
    return e_f  * (s+1)*(s+2) * ( 1 - 1j * (t - R + 1j*z0)/s )**(-s-3) * (1j/s)**2 + \
            e_f  * (s+1)*(s+2) * ( 1 - 1j * (t + R + 1j*z0)/s )**(-s-3) * (1j/s)**2





def E_theta(t, x, y, z):
	return 0

def B_r(t, x, y, z):
	return 0

def B_z(t, x, y, z):
	return 0

def E_r_complex(t, x, y, z):
    r = 1.0 / R_tilde(t, x, y, z)
    return A0 * 3.0/2 * sin_2_theta(t, x, y, z) *r * ( G_0_neg(t, x, y, z) * r**2 + \
            G_1_neg(t, x, y, z)*r + G_2_neg(t, x, y, z)/3.0 )
def E_z_complex(t, x, y, z):
    r = 1.0 / R_tilde(t, x, y, z)
    return B0 *r * (  (3 * cos_theta(t, x, y, z)**2 - 1)*r * (G_0_neg(t, x, y, z) *r +
            G_1_pos(t, x, y, z) ) -   sin_theta(t, x, y, z)**2 * G_2_neg(t, x, y, z)  )
def B_theta_complex(t, x, y, z):
    r = 1.0 / R_tilde(t, x, y, z)
    return C0 * sin_theta(t, x, y, z)*r * ( G_1_neg(t, x, y, z)*r +  G_2_pos(t, x, y, z) )

##### NEW
# def E_r_complex(t, x, y, z):
#     return 3.0/4 * A0 * np.exp(-z0 + 1j*t) * scipy.special.jn(2,R_tilde(t, x, y, z)) * sin_2_theta(t, x, y, z)
#
# def E_z_complex(t, x, y, z):
#     return B0 * np.exp(-z0 + 1j*t) *( scipy.special.jn(0,R_tilde(t, x, y, z)) + \
#         scipy.special.jn(2,R_tilde(t, x, y, z)) * 1.0/4 * (1+3*cos_2_theta(t, x, y, z)) )
#
# def B_theta_complex(t, x, y, z):
#     return 3.0/2 * 1j * C0 * np.exp(-z0 + 1j*t) * scipy.special.jn(1,R_tilde(t, x, y, z)) * sin_theta(t, x, y, z)



def E_r(t, x, y, z):
	return E_r_complex(t, x, y, z).real
def E_z(t, x, y, z):
	return E_z_complex(t, x, y, z).real
def B_theta(t, x, y, z):
    return B_theta_complex(t, x, y, z).real

'''
===============================================
NEW
'''
def E_r_new_complex(t, x, y, z):
    return 3.0/4 * A0 * np.exp(-z0 + 1j*t + 1j*phi_0) * scipy.special.jn(2,R_tilde(t, x, y, z)) * sin_2_theta(t, x, y, z)

def E_z_new_complex(t, x, y, z):
    return B0 * np.exp(-z0 + 1j*t+ 1j*phi_0) *( scipy.special.jn(0,R_tilde(t, x, y, z)) + \
        scipy.special.jn(2,R_tilde(t, x, y, z)) * 1.0/4 * (1+3*cos_2_theta(t, x, y, z)) )

def B_theta_new_complex(t, x, y, z):
    return 3.0/2 * 1j * C0 * np.exp(-z0 + 1j*t+ 1j*phi_0) * scipy.special.jn(1,R_tilde(t, x, y, z)) * sin_theta(t, x, y, z)

def E_r_new(t, x, y, z):
	return E_r_new_complex(t, x, y, z).real
def E_z_new(t, x, y, z):
	return E_z_new_complex(t, x, y, z).real
def B_theta_new(t, x, y, z):
    return B_theta_new_complex(t, x, y, z).real


'''
===============================================
'''





def angle_real(t, x, y, z):
    return np.arctan2(y, x)
def cos_real(t, x, y, z):
    return np.cos( angle_real(t, x, y, z) )
def sin_real(t, x, y, z):
    return np.sin( angle_real(t, x, y, z) )
# Transfer Coordinate from Cylindrical to Cartesian system
def E_x(t, x, y, z):
	return cos_real(t, x, y, z) * E_r(t, x, y, z)
def E_y(t, x, y, z):
	return sin_real(t, x, y, z) * E_r(t, x, y, z)
def B_x(t, x, y, z):
	return (-1.0) * sin_real(t, x, y, z) * B_theta(t, x, y, z)
	# return (-1.0) * sin_real(t, x, y, z) * B_theta_new(t, x, y, z)
def B_y(t, x, y, z):
	return (1.0) * cos_real(t, x, y, z) * B_theta(t, x, y, z)
	# return (1.0) * cos_real(t, x, y, z) * B_theta_new(t, x, y, z)
