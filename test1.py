# -*- coding: utf-8 -*-

import numpy as np

############### Test 1 Beam (Linearly Polarized EM wave) A = a0*cos(t−x)y ###############
# c = 3.0E08  #speed_of_light
# k0 = 1E7
# w0 = c*k0
# epsilon = -1  #direction, negative move to left, and positive move to right
# L = 3E-14 * w0
# Zr = 5E-4 * k0
name = "Linearly Polarized EM wave A = a0*cos(t−x)y"

a0 = 0.2

def E_x(t, x, y, z):
	return 0.0

def E_y(t, x, y, z):
	return a0 * np.sin(t - x) + y*0 + z*0

def E_z(t, x, y, z):
	return 0 #test

def B_x(t, x, y, z):
	return 0

def B_y(t, x, y, z):
	return 0

def B_z(t, x, y, z):
	return a0 * np.sin(t - x) + y*0 + z*0
