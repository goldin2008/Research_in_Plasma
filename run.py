from test1 import *
from pylab import *
import numpy as np
from scipy.integrate import odeint
from math import *
# from matplotlib import pyplot as plt


############### RUN Dynamic Equation ###############
q_over_m = -1
c = 3.0E08  #speed_of_light
k0 = 1E7
w0 = c*k0
L = 3E-14 * w0
L = 20
r = 1E-5 * k0 #beam waist

print 'Name of Beam: ', name
# print "k0 = ", k0
# print "w0 = ", w0
# print "L_tao = ", L
# print "Zr = ", Zr
# print "r = ", r
# print 'a0 = ', a0


def traj(y_vec, t):
    px, py, pz, x, y, z= y_vec  # tuple unpacking

    return [
    q_over_m*( E_x(t, x, y, z) + ( py/(1+px**2+py**2+pz**2)**(1.0/2)*B_z(t, x, y, z) -\
    pz/(1+px**2+py**2+pz**2)**(1.0/2)*B_y(t, x, y, z) ) ),

    q_over_m*( E_y(t, x, y, z) + ( pz/(1+px**2+py**2+pz**2)**(1.0/2)*B_x(t, x, y, z) -\
    px/(1+px**2+py**2+pz**2)**(1.0/2)*B_z(t, x, y, z) ) ),

    q_over_m*( E_z(t, x, y, z) + ( px/(1+px**2+py**2+pz**2)**(1.0/2)*B_y(t, x, y, z) -\
    py/(1+px**2+py**2+pz**2)**(1.0/2)*B_x(t, x, y, z) ) ),

    px/(1+px**2+py**2+pz**2)**(1.0/2),
    py/(1+px**2+py**2+pz**2)**(1.0/2),
    pz/(1+px**2+py**2+pz**2)**(1.0/2)
    ]


n = 51
t_start = 0*2.0*pi
t_final = 30*2.0*pi
delta_t = 2.0*pi/n
num_steps = (t_final -t_start)/delta_t

number = (t_final - t_start)/delta_t

t = np.arange(t_start, t_final, delta_t)

x = np.linspace(-2.5*L, 2.5*L, 201)#np.arange(-r, r, r/100)
y = np.linspace(-2.5*L, 2.5*L, 201)#np.arange(-r, r, r/100)
z = np.linspace(-5*r, 5*r, 201)




X,Z = meshgrid(x, z) # grid of point

F = E_y(0, X, 0, Z )

############### Figure 4 ###############
plt.figure(4)
# im = imshow(F,cmap=cm.RdBu) # drawing the function
subplot(111)
im = imshow(flipud(transpose(F)), cmap=matplotlib.cm.gist_rainbow, extent=(z.min(), z.max(), x.min(), x.max()))
colorbar(im)
title('$E_y(t, x, y=0, z)$')

# subplot(212)
# F1 = B_x(0, X, 0, Z )
# im1 = imshow(flipud(transpose(F1)), cmap=matplotlib.cm.gist_rainbow, extent=(z.min(), z.max(), x.min(), x.max()))
# colorbar(im1)
# title('$B_x(t, x, y=0, z)$')



# Choose random starting points, uniformly distributed from -15 to 15
N_particles = 1

np.random.seed(1)
# x0 = -r/4 + r/2 * np.random.random((N_particles, 3))
# x0_1 = -r/4 + r/2 * np.random.random((N_particles, 2))
x0_1 = 0.0*np.ones((N_particles, 2))
x0_2 = -40.0*np.ones((N_particles, 1))
x0 = np.concatenate((x0_1, x0_2), axis=1)

print 'x0 = ', x0

p0 = np.zeros((N_particles, 3))
p0_1 = np.zeros((N_particles, 2))
p0_2 = 10.0*np.ones((N_particles, 1))
p0 = np.concatenate((p0_1, p0_2), axis=1)
print 'p0 = ', p0

y0 = np.concatenate((p0, x0), axis=1)
print 'y0 = ', y0

y_result = np.asarray( [odeint(traj, y0i, t) for y0i in y0] )

for i in range( N_particles ):
	px, py, pz, x, y, z = y_result[:i+1].T
# 	print 'len(z) = ', len(z)
	#create array
	z_ave = np.ones(len(z))
	z_ave = np.array(z_ave)

	z_array = np.ones(len(z))
	z_array = np.array(z_array)

	z = z.T
	z = z[0]
# 	print 'z----', z[0]

############### Figure 1 ###############

	plt.figure(1)
	title("Trajectory")

	for j in range( len(z) ):
		z_array[j] = z[j]

		if j < n/2:
			z_array[j] = z[j]
		elif j > len(z)-n/2:
			z_array[j] = z[j]
		else:
			z_array[j] = z[j-n/2:j+n/2+1].sum()/n

	a = len(t)
# 	print ' length of z_ave[n/2:a-n/2] = ', len(z_array[n/2:a-n/2])
	subplot(231)
# 	plt.plot( z[n/2:a-n/2]-z_array[n/2:a-n/2], x[n/2:a-n/2], 'b.')
	plt.plot( z[n/2:a-n/2]-z_array[n/2:a-n/2], x[n/2:a-n/2], '-')
	plt.xlabel("z")
	plt.ylabel("x")

	subplot(232)
# 	plt.plot( z[n/2:a-n/2]-z_array[n/2:a-n/2] , y[n/2:a-n/2], 'b.')
	plt.plot( z[n/2:a-n/2]-z_array[n/2:a-n/2] , y[n/2:a-n/2], '-')
	plt.xlabel("z")
	plt.ylabel("y")

	subplot(233)
# 	plt.plot( z[n/2:a-n/2]-z_array[n/2:a-n/2] , pz[n/2:a-n/2], 'b.')
	plt.plot( z[n/2:a-n/2]-z_array[n/2:a-n/2] , pz[n/2:a-n/2], '-')
	plt.xlabel("z")
	plt.ylabel("pz")
# 	# plt.title("$\delta = \sqrt{1/2}$")

	subplot(234)
	plt.plot( t[n/2:a-n/2], z[n/2:a-n/2], '-')
	plt.xlabel("t")
	plt.ylabel("z")

	subplot(235)
	plt.plot( t[n/2:a-n/2] , z_array[n/2:a-n/2], '-')
	plt.xlabel("t")
	plt.ylabel("z_ave")


	subplot(236)
	plt.plot( t[n/2:a-n/2] , z[n/2:a-n/2]-z_array[n/2:a-n/2], '-')
	plt.xlabel("t")
	plt.ylabel("z-z_ave")

############### Figure 2 ###############
	plt.figure(2)
	subplot(321)
	plt.plot( t , px)
	plt.xlabel("Time-axis")
	plt.ylabel("Px-axis")
	plt.title("Momentum L = 20")

	subplot(322)
	plt.plot( t, x)
	plt.xlabel("Time-axis")
	plt.ylabel("X-axis")
	plt.title("Coordinate")

	subplot(323)
	plt.plot( t, py)
	plt.xlabel("Time-axis")
	plt.ylabel("Py-axis")

	subplot(324)
	plt.plot( t, y)
	plt.xlabel("Time-axis")
	plt.ylabel("Y-axis")

	subplot(325)
	plt.plot( t, pz)
	plt.xlabel("Time-axis")
	plt.ylabel("Pz-axis")

	subplot(326)
	plt.plot( t, z)
	plt.xlabel("Time-axis")
	plt.ylabel("Z-axis")


############### Radiation Spectrum ###############
# Calculate on-axis spectrum
vx = px/(1+px**2+py**2+pz**2)**(1.0/2)
vy = py/(1+px**2+py**2+pz**2)**(1.0/2)
vz = pz/(1+px**2+py**2+pz**2)**(1.0/2)


w_start = 0.0
w_final = 1000.0
delta_w = 2.0

num_w = np.floor((w_final -w_start)/delta_w)+1

def my_range(start, end, step):
	  while start <= end:
	           yield start
	           start += step

#Create arrays for w and s with num_w elements
w = np.zeros(num_w)
w = np.array(w)
s = np.zeros(num_w)
s = np.array(s)

j=0

for w_i in my_range(w_start, w_final, delta_w):
	w[j] = w_start+j*delta_w
	vx_sin = 0.0
	vy_sin = 0.0
	vx_cos = 0.0
	vy_cos = 0.0

	# cumulatively sum the areas
	for i in range(1, int(num_steps)):
		vx_sin += delta_t * ( vx[i-1] * sin( w_i * ( t[i-1] - z[i-1] ) ) + vx[i] * sin( w_i * ( t[i] - z[i] ) ) )/2.0
		vy_sin += delta_t * ( vy[i-1] * sin( w_i * ( t[i-1] - z[i-1] ) ) + vy[i] * sin( w_i * ( t[i] - z[i] ) ) )/2.0
		vx_cos += delta_t * ( vx[i-1] * cos( w_i * ( t[i-1] - z[i-1] ) ) + vx[i] * cos( w_i * ( t[i] - z[i] ) ) )/2.0
		vy_cos += delta_t * ( vy[i-1] * cos( w_i * ( t[i-1] - z[i-1] ) ) + vy[i] * cos( w_i * ( t[i] - z[i] ) ) )/2.0

	s[j] = 1.0/(4*pi**2) * ( ( vx[num_steps-1]/(1-vz[num_steps-1]) - vx[0]/(1-vz[0]) + vx_sin )**2 +\
							 ( vy[num_steps-1]/(1-vz[num_steps-1]) - vy[0]/(1-vz[0]) + vy_sin )**2 + ( vx_cos )**2 + ( vy_cos )**2 )

	j +=1


############### Figure 3 ###############
plt.figure(3)
ax = subplot(111)
ax.plot( w, s)
ax.set_xlim(w.min(), w.max())
ax.set_ylim(s.min(), s.max())
ax.set_xlabel('w/wp')
ax.set_ylabel('on-axis spectrum')
# # Horizontal black line at y=0
# ax.axhline(y=0.001, color='b')
# # Vertical black line at y=0
ax.axvline(x=400, linewidth=1, color='r', linestyle='--', label = 'w/wp = 400')
ax.legend()
############### Radiation End ###############

plt.show()
