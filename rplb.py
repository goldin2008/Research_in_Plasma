#!/usr/bin/env python
# -*- coding: utf-8 -*-
### Laser Module
# Define some variables:
from math import *
import numpy as np

beam = 100
name = "Radically Polorized Laser Beam"
A0 = 1
z0 = 350
phi_0 = 0

# R Tilde radius
def R_tilde_radius(t, x, y, z):
	return ( (x**2+y**2+z**2-z0**2)**2 + (2*z*z0)**2 )**(1.0/2)

# angle α
def alpha(t, x, y, z):
	return np.arctan( z0/z )

# angle β
def beta(t, x, y, z):
	return np.arctan( 2.0*z*z0 / (x**2+y**2+z**2-z0**2) )

# angle includes sin
def angle_sin(t, x, y, z):
	return np.sin( beta(t, x, y, z)/2 ) * R_tilde_radius(t, x, y, z)**(1.0/2)

# angle includes cos
def angle_cos(t, x, y, z):
	return np.cos( beta(t, x, y, z)/2 ) * R_tilde_radius(t, x, y, z)**(1.0/2)

# Bessel Function
def j0_real(t, x, y, z):
	return 1.0/ ( 2* R_tilde_radius(t, x, y, z)**(1.0/2) ) *(
	np.exp( -1.0* angle_sin(t, x, y, z) ) *
	np.cos( angle_cos(t, x, y, z) - pi/2 - beta(t, x, y, z)/2 )
	-
	np.exp( 1.0* angle_sin(t, x, y, z) ) *
	np.cos( angle_cos(t, x, y, z) + pi/2 + beta(t, x, y, z)/2 )
	)
def j0_im(t, x, y, z):
	return 1.0/ ( 2* R_tilde_radius(t, x, y, z)**(1.0/2) ) *(
	np.exp( -1.0* angle_sin(t, x, y, z) ) *
	np.sin( angle_cos(t, x, y, z) - pi/2 - beta(t, x, y, z)/2 )
	+
	np.exp( 1.0* angle_sin(t, x, y, z) ) *
	np.sin( angle_cos(t, x, y, z) + pi/2 + beta(t, x, y, z)/2 )
	)

def j1_real(t, x, y, z):
	return 1.0/ ( 2* R_tilde_radius(t, x, y, z) ) *(
	np.exp( -1.0* angle_sin(t, x, y, z) ) *
	np.cos( angle_cos(t, x, y, z) - pi/2 - beta(t, x, y, z) )
	-
	np.exp( 1.0* angle_sin(t, x, y, z) ) *
	np.cos( angle_cos(t, x, y, z) + pi/2 + beta(t, x, y, z) )
	)\
	- 1.0/ ( 2* R_tilde_radius(t, x, y, z)**(1.0/2) ) *(
	np.exp( -1.0* angle_sin(t, x, y, z) ) *
	np.cos( angle_cos(t, x, y, z) - beta(t, x, y, z)/2 )
	+
	np.exp( 1.0* angle_sin(t, x, y, z) ) *
	np.cos( angle_cos(t, x, y, z) + pi/2 + beta(t, x, y, z)/2 )
	)
def j1_im(t, x, y, z):
	return 1.0/ ( 2* R_tilde_radius(t, x, y, z) ) *(
	np.exp( -1.0* angle_sin(t, x, y, z) ) *
	np.sin( angle_cos(t, x, y, z) - pi/2 - beta(t, x, y, z) )
	+
	np.exp( 1.0* angle_sin(t, x, y, z) ) *
	np.sin( angle_cos(t, x, y, z) + pi/2 + beta(t, x, y, z) )
	)\
	- 1.0/ ( 2* R_tilde_radius(t, x, y, z)**(1.0/2) ) *(
	np.exp( -1.0* angle_sin(t, x, y, z) ) *
	np.sin( angle_cos(t, x, y, z) - beta(t, x, y, z)/2 )
	-
	np.exp( 1.0* angle_sin(t, x, y, z) ) *
	np.sin( angle_cos(t, x, y, z) + pi/2 + beta(t, x, y, z)/2 )
	)

def j2_real(t, x, y, z):
	return 3.0/ R_tilde_radius(t, x, y, z)**(1.0/2) *(
	np.cos(beta(t, x, y, z)/2) * j1_real(t, x, y, z)
	+
	np.sin(beta(t, x, y, z)/2) * j1_im(t, x, y, z)
	) - j0_real(t, x, y, z)
def j2_im(t, x, y, z):
	return 3.0/ R_tilde_radius(t, x, y, z)**(1.0/2) *(
	np.cos(beta(t, x, y, z)/2) * j1_im(t, x, y, z)
	-
	np.sin(beta(t, x, y, z)/2) * j1_real(t, x, y, z)
	) - j0_im(t, x, y, z)

# sin, cos function part
def sin_theta_real(t, x, y, z):
	return (x**2+y**2)**(1.0/2) / R_tilde_radius(t, x, y, z)**(1.0/2) * np.cos(beta(t, x, y, z)/2)
def sin_theta_im(t, x, y, z):
	return (x**2+y**2)**(1.0/2) / R_tilde_radius(t, x, y, z)**(1.0/2) * (-1)*np.sin(beta(t, x, y, z)/2)

def sin_2_theta_real(t, x, y, z):
	return 2 * (x**2+y**2)**(1.0/2) * (z**2+z0**2)**(1.0/2) / \
		 R_tilde_radius(t, x, y, z) * np.cos( alpha(t, x, y, z) - beta(t, x, y, z) )
def sin_2_theta_im(t, x, y, z):
	return 2 * (x**2+y**2)**(1.0/2) * (z**2+z0**2)**(1.0/2) / \
		 R_tilde_radius(t, x, y, z) * np.sin( alpha(t, x, y, z) - beta(t, x, y, z) )

# P_2 function
def p2_cos_real(t, x, y, z):
	return 1 - 3.0/2* (x**2+y**2) / R_tilde_radius(t, x, y, z) * np.cos(beta(t, x, y, z))
def p2_cos_im(t, x, y, z):
	return 1 + 3.0/2* (x**2+y**2) / R_tilde_radius(t, x, y, z) * np.sin(beta(t, x, y, z))

# Time term part
def time_real(t, x, y, z):
	return np.cos(t + phi_0)
def time_im(t, x, y, z):
	return np.sin(t + phi_0)


def E_r(t, x, y, z):
	return A0 * np.exp(-z0) * (
	time_real(t, x, y, z) * j2_real(t, x, y, z) * sin_2_theta_real(t, x, y, z)
	-
	time_real(t, x, y, z) * j2_im(t, x, y, z) * sin_2_theta_im(t, x, y, z)
	-
	time_im(t, x, y, z) * j2_real(t, x, y, z) * sin_2_theta_im(t, x, y, z)
	-
	time_im(t, x, y, z) * j2_im(t, x, y, z) * sin_2_theta_real(t, x, y, z)
	)

def E_z(t, x, y, z):
	return 4.0/3 * A0 * np.exp(-z0) * (
	time_real(t, x, y, z) * j0_real(t, x, y, z)
	-
	time_im(t, x, y, z) * j0_im(t, x, y, z)
	+
	time_real(t, x, y, z) * j2_real(t, x, y, z) * p2_cos_real(t, x, y, z)
	-
	time_real(t, x, y, z) * j2_im(t, x, y, z) * p2_cos_im(t, x, y, z)
	-
	time_im(t, x, y, z) * j2_real(t, x, y, z) * p2_cos_im(t, x, y, z)
	-
	time_im(t, x, y, z) * j2_im(t, x, y, z) * p2_cos_real(t, x, y, z)
	)

def B_theta(t, x, y, z):
    return 2.0 / 1 * A0 * np.exp(-z0) * (
	time_real(t, x, y, z) * j1_real(t, x, y, z) * sin_theta_real(t, x, y, z)
	-
	time_real(t, x, y, z) * j1_im(t, x, y, z) * sin_theta_im(t, x, y, z)
	-
	time_im(t, x, y, z) * j1_real(t, x, y, z) * sin_theta_im(t, x, y, z)
	-
	time_im(t, x, y, z) * j1_im(t, x, y, z) * sin_theta_real(t, x, y, z)
	)

# Transfer Coordinate from Cylindrical to Cartesian system
def E_x(t, x, y, z):
	return x / (x**2+y**2)**(1.0/2) * E_r(t, x, y, z)
def E_y(t, x, y, z):
	return y / (x**2+y**2)**(1.0/2) * E_r(t, x, y, z)
def B_x(t, x, y, z):
	return (-1.0) * y / (x**2+y**2)**(1.0/2) * B_theta(t, x, y, z)
def B_y(t, x, y, z):
	return (1.0) * x / (x**2+y**2)**(1.0/2) * B_theta(t, x, y, z)
def B_z(t, x, y, z):
	return 0

# class Gauss:
# 	title = 'Gauss'
#
# 	def _init_(self, t, z):
# 		self.time = t
# 		self.z_axis = z
#
# 	def ZETA(t, z):
# 		return z - t
