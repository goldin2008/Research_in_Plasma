import numpy as np

############### Gaussian Beam ###############

name = "Gaussian Beam"
c = 3.0E08  #speed_of_light
k0 = 1E7
w0 = c*k0
epsilon = -1  #direction, negative move to left, and positive move to right
L = 3E-14 * w0
L = 20
Zr = 5E-4 * k0
a0 = 1.25
r = 1E-5 * k0 #beam waist


def ZETA(t, z):
	return epsilon * t - z

def f_exp(t, x, y, z):
	return np.exp(-(x**2+y**2)*Zr/(2*(z**2+Zr**2)) - ZETA(t,z)**2/L**2)

def f_sin(t, x, y, z):
	return np.sin((x**2+y**2)*z/(2*(z**2+Zr**2)) + (z-t*epsilon) - np.arctan(z/Zr))

def f_cos(t, x, y, z):
	return np.cos((x**2+y**2)*z/(2*(z**2+Zr**2)) + (z-t*epsilon) - np.arctan(z/Zr))

def A_x(t, x, y, z):
	return Zr/(Zr**2 + z**2)**(1.0/2) * f_exp(t, x, y, z) * f_cos(t, x, y, z)


def E_x(t, x, y, z):
	return a0*f_exp(t, x, y, z) * epsilon *\
	( 2*ZETA(t,z) * f_cos(t, x, y, z) -\
	L**2*f_sin(t, x, y, z) ) / ( L**2*np.sqrt(1+z**2/Zr**2) )

def E_y(t, x, y, z):
	return 0.0

def E_z(t, x, y, z):
	return a0*f_exp(t, x, y, z) * x * epsilon *\
	( (L**2*Zr - 2*z*ZETA(t,z)) * f_cos(t, x, y, z) +\
	(L**2*z + 2*Zr*ZETA(t,z)) * f_sin(t, x, y, z) ) / ( L**2*np.sqrt(1+z**2/Zr**2)*(z**2+Zr**2) )

def B_x(t, x, y, z):
	return a0*f_exp(t, x, y, z) *x*y * Zr *\
	( 2*z*Zr * f_cos(t, x, y, z) +\
	(z**2 - Zr**2) * f_sin(t, x, y, z)  ) / (z**2+Zr**2)**(5.0/2)

def B_y(t, x, y, z):
	return a0*f_exp(t, x, y, z) * Zr *\
	( ( 2* L**2 * (-x**2+y**2)*z*Zr + 4*(z**2+Zr**2)**2 * ZETA(t,z) ) * f_cos(t, x, y, z) -\
	L**2*( x**2 * (z**2-Zr**2)+ y**2 * (-1*z**2+Zr**2) + 2 * (z**2+Zr**2)**2 ) *\
	f_sin(t, x, y, z)  )/( 2*L**2 *(z**2+Zr**2)**(5.0/2) )

def B_z(t, x, y, z):
	return a0*f_exp(t, x, y, z) * y * Zr *\
	( Zr * f_cos(t, x, y, z) +\
	z * f_sin(t, x, y, z)  ) / (z**2+Zr**2)**(3.0/2)
