#!/usr/bin/python
import numpy as np
from matplotlib import pyplot as plt
from math import *

from pylab import *

import time

from mpl_toolkits.mplot3d import Axes3D

from prl import *

########################## CONTINUE #############################################
########################### TESTING PART ############################################
m = 9.1E-31
c = 3.0E+8  #speed_of_light
k0 = 2 * pi / (800 * 1E-9)
w0 = c*k0

'''
Copy the following statements from run.py
'''

'''
Test Case 1
'''
#num = 768
#N_particles_per_job = int(1E3 *27 * num )


'''
Test Case 2
'''
#num = 864
#N_particles_per_job = int(1E4 *4 * num )

'''
Test Case 3
'''
# num = 768
# N_particles_per_job = int(1E3 *6**3 * num )

'''
Test
'''
#num = 864
num = 12*1*36


#N_slice = 40000
N_slice = 50*1*50
N_particles = N_slice * num

t_start = -20E-15 * w0
t_final = (120)*1E-15 * w0
t_number = 500



delta_t = (t_final - t_start)/ t_number
t = np.arange(t_start, t_final, delta_t)
number_of_time_step = len(t)


'''
Initial Time
'''
t_i = time.clock()


'''
    ######################## TEST ########################
    '''
##for index in xrange(3):
##    print index
#index = np.arange(12*36)
##x = index // (1*36) % 12
##z = index // (1*12)
#x = index % 12
#z = index // 12
#
#print index
##print x
##print z
#
##plt.figure()
##plt.plot(x, z, 'bo')
##plt.xlabel('$x $')
##plt.ylabel('$z $')
##plt.show()
#
#print np.linspace(2.0, 3.0, num=5, endpoint=False)
#
#exit()

'''
    ######################## END TEST ########################
    '''



'''
######################## Figures For Energy ########################
'''
'''
######################## Read Data ########################
'''

x_start = np.empty(N_particles)
y_start = np.empty(N_particles)
z_start = np.empty(N_particles)

x_final = np.empty(N_particles)
y_final = np.empty(N_particles)
z_final = np.empty(N_particles)

energy = np.empty(N_particles)

def load_into(name, data, scale=None):
    def mkname(i):
        return 'input_file/%s%d.npy' % (name, i)

    print name
    loc = 0
    for i in xrange(num):
        f = mkname(i)
        data[loc:loc+N_slice] = np.load(mkname(i))
        loc += N_slice
    if scale:
        data *= scale
'''
for n,d in ('start_x', x_start),('start_y', y_start), ('start_z', z_start),\
           ('final_x', x_final),('final_y', y_final), ('final_z', z_final),\
           ('energy', energy):
    load_into(n, d)

x_start /= 2*pi
y_start /= 2*pi
z_start /= 2*pi
x_final /= 2*pi
y_final /= 2*pi
z_final /= 2*pi
'''

'''
Load data into the empty array
'''
for n,d,s in ('start_x', x_start, 1.0/(2*pi)),('start_y', y_start, 1.0/(2*pi)),\
             ('start_z', z_start, 1.0/(2*pi)), ('energy', energy, None),\
                 ('final_x', x_final, 1.0/(2*pi)):
    load_into(n, d, s)

print 'energy max in ALL = ', max(energy)
print 'energy min in ALL = ', min(energy)
print 'energy length in ALL = ', len(energy)




'''
######################## Filter Data ########################
'''
#idx = np.where( (energy >= 0)  )
#idx = np.where((x_start >= 0) & (x_start <= 1.5) & (y_start == 0) & (z_start == 0)  )
idx = np.where((x_start >= 0) & (x_start <= 1.5) & (y_start == 0)  )
#idx = np.where((x_start >= 0) & (x_start <= 1.5) & (y_start == 0) & (energy <= 1.2E3) )
#idx = np.where((x_start >= 0) & (x_start <= 1.5) & (y_start == 0) & (energy <= 810) )
#idx = np.where((energy == inf))
# idx = np.where((y_start == 0))
# print 'shape: ', x_start[idx].shape, x_start.shape

x_sel = x_start[idx]
z_sel = z_start[idx]
energy_sel = energy[idx]
#energy_sel = x_final[idx]
y_sel = y_start[idx]
x_f = x_final[idx]
print 'len of x_sel', len(x_sel)
print 'len of z_sel', len(z_sel)
print 'len of y_sel', len(y_sel)
print 'len of energy_sel', len(energy_sel)

# x_sel = x_start
# z_sel = z_start
# energy_sel = energy

print 'energy_for_matrix max = ', max(energy_sel)
print 'energy_for_matrix min = ', min(energy_sel)
print 'energy_for_matrix len = ', len(energy_sel)


'''
    ######################## Figure  ########################
    3D Initial Positions of Particles
    '''

#fig = plt.figure(1)
#ax = Axes3D(fig)
## for i in range(1, 800):
##         file_name = str(i) + '.npz'
##         data = np.load( 'initial_position/'+file_name )
##         data = data['arr_0']
##         p = data
##         ax.plot(p[:,0], p[:,1], p[:,2], 'o')
##         # p = np.concatenate((p, data), axis=0)
#ax.plot(x_sel, y_sel, z_sel, 'o')
#
#ax.set_xlabel('X label')
#ax.set_ylabel('Y label')
#ax.set_zlabel('Z label')
#
## plt.show()
#plt.savefig('figure/position.png')
#
#
#plt.figure(2)
#plt.plot(x_sel, energy_sel, 'bo')
#plt.xlabel('$x $')
#plt.ylabel('$Energy $')
## plt.savefig('fig/field.png')
#plt.savefig('figure/field.pdf')
#
#
#exit()

'''
######################## Figure 11-16 ########################
Energy Distribution
'''

#dx = 4.0/(24*200)
#dz = 4.0/(24*200)
dx = 2.0/(12*50)
dz = 6.0/(36*50)

def x_coord(x):
#    return np.array((x + 2)/dx + 0.5, dtype='int')
    return np.array((x + 0.0)/dx + 0.5, dtype='int')

def z_coord(z):
    return np.array((z + 2)/dz + 0.5, dtype='int')

xc = x_coord(x_sel)
zc = z_coord(z_sel)

#a = [-2, -2+4.0/(24*200), -2+4.0/(24*200)*2, -2+4.0/(24*200)*3 ]
#a = np.array(a)
# print 'dx = ', dx
# print 'a = ', a
# print 'a = ', a+2
# print 'a = ', np.array((a + 2)/dx, dtype='int')
# print 'a = ', np.array((a + 2)/dx+0.5, dtype='int')

print 'Before: '
print 'xc sel index in matrix = ', xc, len(xc)
print 'zc sel index in matrix = ', zc, len(zc)

x_off = xc.min()
z_off = zc.min()

xc -= x_off
zc -= z_off

print 'After: '
print 'xc sel index in matrix = ', xc, len(xc)
print 'zc sel index in matrix = ', zc, len(zc)

nx = xc.max() - xc.min() + 1
nz = zc.max() - zc.min() + 1

print nx, nz, xc.max(), zc.max()
matrix = np.zeros((nx, nz))
#matrix = -2*np.ones((nx, nz))

#matrix = np.zeros((len(xc), len(xc)))

print 'len of matrix', len(matrix), len(matrix[0]), len(matrix)*len(matrix[0])
#exit()

for k in xrange(len(xc)):
    matrix[xc[k], zc[k]] = energy_sel[k]
#    matrix[xc[k], zc[k]] = math.log10( math.fabs(energy_sel[k])+1 )


print 'matrix shape: ', matrix.shape

#exit()

im = imshow(matrix, aspect='auto',origin='lower', cmap=matplotlib.cm.gist_rainbow, extent=( z_sel.min(), z_sel.max(), x_sel.min(), x_sel.max()))
colorbar(im)

plt.title("$E \, (keV)$ and $\phi$ = " + str(phi_0/pi) + "PI" ) 
#plt.title("$log_{10}(|x_f|+1) \, $ and $\phi$ = " + str(phi_0/pi) + "PI" )
plt.xlabel("$z_0/\lambda_0$")
plt.ylabel("$x_0/\lambda_0$")
#plt.savefig('figure/f1-new.pdf')
plt.savefig('figure/f1-new.png')
# plt.show()

#xu = sorted(np.unique(xc))
#zu = np.unique(zc)
#print len(xu), len(zu)
#print nx, nz
#k = 0
#for i in range(len(xu)):
#    if xu[i] != k:
#        print i
#        k += 2
#    else:
#        k += 1
#
#print 'Job is Done.'

exit()

'''
Final Time
'''
t_f = time.clock()
t_cost = t_f - t_i

# exit()

print 'Start Time = ', t_start
print 'End Time = ', t_final
print 'Time Step = ', delta_t
print 'Number of Time Interval = ', t_number

print 'Number of Particles AT BEGINNING  = ', N_particles
print 'Computing and Plotting Cost Time = ', t_cost
# print "Number of Used Particles BETWEEN r'y = $-\lambda_0$ AND $\lambda_0$' (r'50 kev $\leq$ energy $\leq$ 500 kev')  = ", sum(hist_E)
# u = 100.0 * sum(hist_E) / N_particles

print 'Maximum Energy chosen = ', max(energy_sel)
print 'Minimum Energy chosen = ', min(energy_sel)

u = len(energy_sel)*100.0/ N_particles
print 'Total Particles = ',  N_particles
print 'Used Particles = ', len(energy_sel)
print ('Utilization of Particles = %.2f %%' % u)
print 'Job is Done.'


'''
######################## SUMMARY OUTPUT ########################
'''

fout = open('figure/SUMMARY.txt', 'w')
fout.write( 'Summary Information: \n\n')
#fout.write( 'Initial x range : %s\n' % str(x_) )
#fout.write( 'Initial y range : %s\n' % str(y_) )
#fout.write( 'Initial z range : %s\n' % str(z_) )
#
#fout.write( 'Final x range : %s\n' % str(x_f) )
#fout.write( 'Final y range : %s\n' % str(y_f) )
#fout.write( 'Final z range : %s\n' % str(z_f) )

fout.write( 'Start Time = %s (fs)\n\n' % str(t_start/w0/1E-15) )
fout.write(  'End Time = %s (fs)\n\n' % str(t_final/w0/1E-15) )
fout.write(  'Time Step = %s (fs)\n\n' % str(delta_t/w0/1E-15) )
fout.write(  'Number of Time Interval = %s \n\n' % str(t_number) )
fout.write( 'Number of Particles AT BEGINNING  = %s \n\n' % str(N_particles) )
fout.write( 'Loading Data and Plotting Cost Time = %s second \n\n' % str(t_cost) )

fout.write( 'energy max in ALL = %s \n\n' % str(max(energy)) )
fout.write( 'energy min in ALL = %s \n\n' % str(min(energy)) )
fout.write( 'energy length in ALL = %s \n\n' % str(len(energy)) )
fout.write( 'Maximum Energy CHOSEN = %s \n\n' % str(max(energy_sel)) )
fout.write( 'Minimum Energy CHOSEN = %s \n\n' % str(min(energy_sel)) )
fout.write( 'Energy Length CHOSEN = %s \n\n' % str(len(energy_sel)) )

# fout.write( 'Number of Used Particles BETWEEN y = - 1/2 * wavelength AND 1/2 * wavelength  (50 Kev <= energy <= 500 Kev)  = %s\n\n' % str( sum(hist_E) ) )
# fout.write( 'Number of Used Particles BETWEEN y = - 1/2 * wavelength AND 1/2 * wavelength  = %s\n\n' % str( sum(hist_E) ) )
#fout.write( "Maximum Energy BETWEEN y = - 1/2 * wavelength AND 1/2 * wavelength = %s\n\n" %str( max(energy_sel) ) )
#fout.write( "Minimum Energy BETWEEN y = - 1/2 * wavelength AND 1/2 * wavelength = %s\n\n" %str( min(energy_sel) ) )

fout.write( 'Total Particles = %u \n\n' % N_particles )
fout.write( 'Used Particles = %u \n\n' % len(energy_sel))
fout.write( 'Utilization of Particles = %.2f %% \n\n' % u)
fout.write( 'Job is Done.' )
fout.close()
