import numpy as np

############### Test 2 Beam (Chapter 9 Note by Prof. Shadwick) ###############
name = "Circularly Polarized EM wave"

a0 = 1.25
d = 0


def E_x(t, x, y, z):
	return -a0*d*np.sin(z-t) + 0*x + 0*y #test 2

def E_y(t, x, y, z):
	return a0* (1-d**2)**(1.0/2)*np.cos(z-t) + 0*x + 0*y #test 2

def E_z(t, x, y, z):
	return 0 #test 2

def B_x(t, x, y, z):
	return -a0* (1-d**2)**(1.0/2)*np.cos(z-t) + 0*x + 0*y #test 2

def B_y(t, x, y, z):
	return -a0*d*np.sin(z-t) + 0*x + 0*y #test 2

def B_z(t, x, y, z):
	return 0 #test 2
