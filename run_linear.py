from __future__ import division

from prl import *
import numpy as np
from scipy.integrate import odeint
from math import *
import time
import sys

# from multiprocessing import Pool


'''
######### Time Start #########
Record all start time
'''
# t_all_start = time.clock()



q = - 1.6E-19
m = 9.1E-31
wave_length = 800 * 1E-9
c = 3.0E+8  #speed_of_light
k0 = 2 * pi / (800 * 1E-9)
w0 = c*k0
T = 2 * pi / w0

# t_start = -20E-15 * w0 # -500E-15, -200E-15, -100E-15, -70E-15, -1E-15, -20E-15
# t_final = 120E-15 * w0 # 100E-15, 1000E-15

'''
Keep the following 4 statements the same as in plot_data.py
'''
#N_particles = int(4E4)
N_particles = int(25)
t_start = -20E-15 * w0
# t_final = (120-1.50540093426)*1E-15 * w0
t_final = (120)*1E-15 * w0
t_number = 100

# N_particles = int( 64 )
# t_start = 1.0
# t_final = 10.0
# t_number = 2

delta_t = (t_final - t_start)/ t_number
t = np.arange(t_start, t_final, delta_t)
# print 't = ', t
'''
print
'''
# print 'Start Time :', t_start
# print 'End Time :', t_final
# print 'Time Step:', delta_t


'''
print
'''
# print 'Name of Beam: ', name


def traj(y_vec, t):
    x, y, z, px, py, pz= y_vec  # tuple unpacking
    # gamma = np.sqrt(1+px**2+py**2+pz**2)
    gi = 1.0 / np.sqrt(1+px**2+py**2+pz**2)
    ex = E_x(t, x, y, z)
    ey = E_y(t, x, y, z)
    ez = E_z(t, x, y, z)
    bx = B_x(t, x, y, z)
    by = B_y(t, x, y, z)
    bz = B_z(t, x, y, z)

    return [
    px * gi,
    py * gi,
    pz * gi,
    ex + ( py * gi * bz - pz * gi * by ) ,
    ey + ( pz * gi * bx - px * gi * bz ) ,
    ez + ( px * gi * by - py * gi * bx )
    ]

'''
######################## Dynamic Equantion Part ########################
Trajectory of Particles
'''

'''
# # # Choose random starting points, uniformly distributed from -15 to 15
'''

'''
print
'''
'''
######################## CREATING DATA ########################
'''
index = int( sys.argv[1] )  #starting postion for Coordinate of each particle

#array_number = 5 * 5 * 32
#array_number = 8 * 8 * 12
#nx = 60
#ny = 60
#nz = 60
# linspace returns array
'''
# Think about cutting the cube along x and y direciton separately
# and count how many sub regions in x-z and y-z planes respectively.
# number 2*6 means how many regions in each x-z plane level
# 1 means the each change in x and y direction
# -1.0 is the Minimum end of the region in x and y direction
# % is Modulus
# Divides left hand operand by right hand operand and returns remainder
# // is Floor Division -
# The division of operands where the result is the quotient in
# which the digits after the decimal point are removed. But if one
# of the operands is negative, the result is floored, i.e., rounded
# away from zero (towards negative infinity)
'''
# position_x = index%(2*2)//2
# position_y = index%(2*2)%2
# position_z = index//(2*2)
#position_x = index%(5*5)//5
#position_y = index%(5*5)%5
#position_z = index//(5*5)

#x_p = np.linspace(-5.0 + position_x * 2, -5.0 + ( position_x + 1 ) * 2 , num=nx)*2*pi
#y_p = np.linspace(-5.0 + position_y * 2, -5.0 + ( position_y + 1 ) * 2 , num=ny)*2*pi

'''
# number 4 means how many regions in each x-y plane level
# 1 means the each change in z direction
# -3.0 is the Minimum end of the region in z direction
'''
#z_p = np.linspace(-14.0 + position_z * 2, -14.0 + ( position_z + 1 ) * 2, num=nz)*2*pi

#array_number = 24 * 1 * 36
#nx = 200
#ny = 1
#nz = 200
array_number = 12 * 1 * 36
nx = 5
ny = 1
nz = 5

#position_x = index%(24*1)%24
#position_y = 0
#position_z = index//(24*1)
#x_p = np.linspace(-2.0 + position_x * (4.0/24), -2.0 + ( position_x + 1 ) * (4.0/24) , num=nx)*2*pi
#y_p = np.linspace(-2.0 + position_y * (4.0/24), -2.0 + ( position_y + 1 ) * (4.0/24) , num=ny)* 0
#z_p = np.linspace(-2.0 + position_z * (6.0/36), -2.0 + ( position_z + 1 ) * (6.0/36) , num=nz)*2*pi
position_x = index % (12*1)
position_y = 0
position_z = index // (12*1)

x_p = np.linspace( 0.0 + position_x * (2.0/12), 0.0 + ( position_x + 1 ) * (2.0/12) , num=nx, endpoint=False)*2*pi
y_p = np.linspace(-2.0 + position_y * (4.0/24), -2.0 + ( position_y + 1 ) * (4.0/24) , num=ny, endpoint=False)* 0
z_p = np.linspace(-2.0 + position_z * (6.0/36), -2.0 + ( position_z + 1 ) * (6.0/36) , num=nz, endpoint=False)*2*pi


# p = np.arange( N_particles * 3 ).reshape(N_particles,3)
p = np.zeros( N_particles * 3 ).reshape(N_particles,3)

for i in range( nx ):
    x = x_p[i]
    # print 'x = ', x
    for j in range( ny ):
        y = y_p[j]
        # print 'y = ', y
        for k in range( nz ):
            z = z_p[k]
            # print 'z = ', z
            p[ i * nz*ny + j * nz + k ] = [x,y,z]
            # print 'p[ i * nz*ny + j * nz + k ] = ', p[ i * nz*ny + j * nz + k ]

momentum = np.zeros((N_particles, 3))
y0 = np.concatenate((p , momentum), axis=1)  # Initial State For Linearly Distribution
# print 'y0 = ', y0
# print 'length of y0 = ', len(y0)

# i = 0
# filename=sys.argv[1]
# with open('x0' + filename ) as f:
#     for line in f:
#         line = line.split()
#         results = map(float, line)
#         results = np.array(results)
#         if line:
#             y0[i] = results
#         i = i + 1

'''
Initial states for all particles, inluding x,y,z,px,py,pz
'''
# y0 = np.concatenate((x0 , p0), axis=1) # Initial State For Random Distribution
# y0 = np.concatenate((p , p0), axis=1)  # Initial State For Linearly Distribution

'''
######################## Multiprocessing via Pool ########################
'''
# def doWork(y0):
#     y_result = np.asarray( [odeint(traj, y0, t)] )
#     y_result = y_result[0] # For each one, we need to grasp it out of []
#     return y_result
#
# t_i = time.clock()
# n_process = 8
# pool = Pool(processes=n_process)
# y_result = pool.map(doWork, y0) # This y_result is list
# y_result = np.asarray(y_result) # Convert list to array
# pool.close()
# pool.join()
# t_f = time.clock()
# t_cost = t_f - t_i
# print 'Multiprocessing via Pool Computing ODE Cost Time = ', t_cost


'''
######################## NON PARALELL ########################
'''
# t_i = time.clock()
y_result = np.asarray( [odeint(traj, y0i, t) for y0i in y0] )
# print 'y_result = ', y_result
# print 'type of y_result = ', type(y_result)
# t_f = time.clock()
# t_cost = t_f - t_i


x, y, z, px, py, pz = y_result[:N_particles].T
# energy of each partile in the all time
energy_all_time = ( np.sqrt( 1+ px**2 + py**2 + pz**2 )-1 )*m*c**2 /1.6E-16
energy = energy_all_time[t_number-1] # energy of each partile in the final time
x_start = x[0]
y_start = y[0]
z_start = z[0]
x_final = x[t_number-1]
y_final = y[t_number-1]
z_final = z[t_number-1]

#final_x = []
#final_y = []
#final_z = []
#final_energy_of_particle = [] # used to collect final energy of the Particles
#                               # satisfying requirments (in the specific region)

# for i in range( N_particles ):
#     if ( y_final[i] >= -0.5 *2*pi and y_final[i] <= 0.5 *2*pi \
#             and x_final[i] >= -2 *2*pi and x_final[i] <= 2 *2*pi \
#             and z_final[i] >= 33 *2*pi and z_final[i] <= 37 *2*pi ):
#         final_energy_of_particle.append( energy[i]  )
#         final_x.append( x_final[i]  )
#         final_y.append( y_final[i]  )
#         final_z.append( z_final[i]  )




# path = '/work/shadwick/yuleinku/crane/linear/save_data/'
# np.save( path + 'final_energy_of_particle'+sys.argv[1], final_energy_of_particle)
# np.save( path + 'final_x'+sys.argv[1], final_x)
# np.save( path + 'final_y'+sys.argv[1], final_y)
# np.save( path + 'final_z'+sys.argv[1], final_z)

#path = '/work/shadwick/yuleinku/crane/linear/save_data/'
path = './local_data/'
np.save( path + 'energy'+sys.argv[1], energy)
np.save( path + 'start_x'+sys.argv[1], x_start)
np.save( path + 'start_y'+sys.argv[1], y_start)
np.save( path + 'start_z'+sys.argv[1], z_start)
np.save( path + 'final_x'+sys.argv[1], x_final)
np.save( path + 'final_y'+sys.argv[1], y_final)
np.save( path + 'final_z'+sys.argv[1], z_final)


'''
PRINT Computing ODE Cost Time
'''
# print 'Computing ODE Cost Time = ', t_cost



'''
######################## CREATING DATA ########################
Use this part when submiting to HCC, otherwise COMMENT this part.
'''

'''
For Local creating results, use fout.open() and fou.close()
'''
# np.save('input_file/'+sys.argv[1], y_result)
# np.savez_compressed('input_file/'+sys.argv[1], y_result)

'''
For HCC output
'''
# path = '/work/shadwick/yuleinku/crane/linear/input_file/'
# np.save( path + sys.argv[1], y_result)





'''
######### Time Final #########
Record all final time
'''
# t_all_final = time.clock()
# t_all_cost = t_all_final - t_all_start
# print 'All Used Time = ', t_all_cost

#exit()


# # fout = open('file/out.txt', 'w')
# for i in range( N_particles ):
#     x, y, z, px, py, pz = y_result[:i+1].T
#     energy = ( np.sqrt( 1+ px**2 + py**2 + pz**2 )-1 )*m*c**2 /1.6E-16
#
#     '''
#     For HCC or Testing, creating results
#     '''
#     # print 'No. \n', i
#     print str(x[:,i])
#     print str(y[:,i])
#     print str(z[:,i])
#     print str(px[:,i])
#     print str(py[:,i])
#     print str(pz[:,i])
#     print str(energy[:,i])
#
#     '''
#     For local creating results and output to new txt file under file folder
#     '''
#     # fout.write( str(x[:,i])+'\n' + \
#     #     str(y[:,i])+'\n' + \
#     #     str(z[:,i])+'\n' + \
#     #     str(px[:,i])+'\n' + \
#     #     str(py[:,i])+'\n' + \
#     #     str(pz[:,i])+'\n' + \
#     #     str(energy[:,i])+'\n' )
# # fout.close()
