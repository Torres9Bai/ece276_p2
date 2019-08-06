# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:42:28 2019

@author: User
"""

import numpy as np
import math
from matplotlib import pyplot as plt
import cv2
import skimage.io as io
import time
from numpy.random import uniform,randn,random


def tic():
  return time.time()
def toc(tstart, name="Operation"):
  print('%s took: %s sec.\n' % (name,(time.time() - tstart)))


if __name__ == '__main__':
  dataset = 21
  
  with np.load("Encoders%d.npz"%dataset) as data:
    encoder_counts = data["counts"] # 4 x n encoder counts
    encoder_stamps = data["time_stamps"] # encoder time stamps

  with np.load("Hokuyo%d.npz"%dataset) as data:
    lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
    lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
    lidar_range_min = data["range_min"] # minimum range value [m]
    lidar_range_max = data["range_max"] # maximum range value [m]
    lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
    
  with np.load("Imu%d.npz"%dataset) as data:
    imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
    imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
  
  with np.load("Kinect%d.npz"%dataset) as data:
    disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
    rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images
  

def lowpassY (xx, dt, RC):
    alpha=dt/(RC+dt)
    y=np.zeros(np.size(xx))
    y[0]=alpha*xx[0]
    for i in range (1, np.size(xx)):
        y[i]=alpha*xx[i]+(1-alpha)*y[i-1]
    return y

imu_dt=0.01
imu_RC=1/(2*math.pi*10)
imu_vyaw=imu_angular_velocity[2,:]
imu_lpvyaw=lowpassY(imu_vyaw,imu_dt,imu_RC)

def C2W(xc,yc,x,y,theta):
    cp=np.array([[x],[y],[1]])
    Tcw=np.array([[math.cos(theta), -math.sin(theta), xc], 
                   [math.sin(theta), math.cos(theta), yc],
                    [0,0,1]])
    w=np.dot(Tcw,cp)
    w=w[0:2,0]
    return w

def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(round(sx))
  sy = int(round(sy))
  ex = int(round(ex))
  ey = int(round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap 

  if dy == 0:
    q = np.zeros((dx+1,1))
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return np.vstack((x,y))

def dist (encoder):
    dL=(encoder[1]+encoder[3])/2*0.0022
    dR=(encoder[0]+encoder[2])/2*0.0022
    dist=(dL+dR)/2
    return dist

def create_particles(x_range, y_range, theta, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = theta
    return particles

def mapCorrelation(im, x_im, y_im, vp, xs, ys):
  '''
  INPUT 
  im              the map 
  x_im,y_im       physical x,y positions of the grid map cells
  vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  
  xs,ys           physical x,y,positions you want to evaluate "correlation" 

  OUTPUT 
  c               sum of the cell values of all the positions hit by range sensor
  '''
  nx = im.shape[0]
  ny = im.shape[1]
  xmin = x_im[0]
  xmax = x_im[-1]
  xresolution = (xmax-xmin)/(nx-1)
  ymin = y_im[0]
  ymax = y_im[-1]
  yresolution = (ymax-ymin)/(ny-1)
  nxs = xs.size
  nys = ys.size
  cpr = np.zeros((nxs, nys))
  for jy in range(0,nys):
    y1 = vp[1,:] + ys[jy] # 1 x 1076
    iy = np.int16(np.round((y1-ymin)/yresolution))
    for jx in range(0,nxs):
      x1 = vp[0,:] + xs[jx] # 1 x 1076
      ix = np.int16(np.round((x1-xmin)/xresolution))
      valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
			                        np.logical_and((ix >=0), (ix < nx)))
      cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
  return cpr


def ptc_motion(x0, y0, theta0, d, vyaw, dt):
    v=d/dt
    if vyaw!=0:
        xcw=x0+v*dt*math.sin(vyaw*dt/2)/(vyaw*dt/2)*math.cos(theta0+vyaw*dt/2)#+np.random.normal(0,0.02)
        ycw=y0+v*dt*math.sin(vyaw*dt/2)/(vyaw*dt/2)*math.sin(theta0+vyaw*dt/2)#+np.random.normal(0,0.02)
    else:
        xcw=x0+v*dt*1*math.cos(theta0+vyaw*dt/2)#+np.random.normal(0,0.02)
        ycw=y0+v*dt*1*math.sin(theta0+vyaw*dt/2)#+np.random.normal(0,0.02)
    position=np.array([xcw, ycw])
    return position

j=0
theta_t=0
theta_w=np.zeros([1,1])
vyaw=np.zeros([1,1])
for i in range (np.size(encoder_stamps)-1):
    t1=encoder_stamps[i]
    t2=encoder_stamps[i+1]
    #theta_t=0
    k=0
    vyawi=0
    while (imu_stamps[j]<t2):
        if j>(np.size(imu_stamps)-2):
            break
        theta_t=theta_t+imu_lpvyaw[j]*(imu_stamps[j+1]-imu_stamps[j])
        k=k+1
        vyawi=vyawi+imu_lpvyaw[j]
        j=j+1
    dt=t2-t1
    theta_w=np.vstack((theta_w,theta_t))
    if k!=0:
        vyawi=vyawi/k
    vyaw=np.vstack((vyaw,vyawi))
    
def L2W(lidar_ranges,lidar_angles,xc,yc,thetac):
    L2C=np.array([[1,0,0.13323],
              [0,1,0],
              [0,0,1]])
    xL=lidar_ranges*math.cos(lidar_angles)
    yL=lidar_ranges*math.sin(lidar_angles)
    p1=np.array([[xL],[yL],[1]])
    p2=np.dot(L2C,p1)
    p3=C2W(xc,yc,p2[0,0],p2[1,0],thetac)
    return p3
    


def predict(particles, dt, d, vyaw, theta):
    #particles[:,2]=particles[:,2]+vyaw*dt
    for i in range (0,particles.shape[0]):
        xi, yi=ptc_motion(particles[i,0], particles[i,1], particles[i,2],d,vyaw,dt)
        particles[i,0]=xi
        particles[i,1]=yi
    particles[:,2]=theta
    return particles
'''
def update(particles, weights, MAP,lidar_ranges,lidar_angles,N):
    occupied = np.log(4)
    free = np.log(1/4)
    occu_max = 20
    occu_min = -20
    best_particle = np.argmax(weights)
    x_c, y_c, theta_c=particles[best_particle, :]
    x_im = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])  # x-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])  # y-positions of each pixel of the map
    Y=np.zeros([2,1])
    for i in range (np.size(lidar_ranges)):
        Yi=L2W(lidar_ranges[i], lidar_angles[i], x_c,y_c,theta_c)
        xi=Yi[0]
        yi=Yi[1]
        Y=np.hstack((Y,np.array([[xi],[yi]])))
    Y=Y[:,1:]
    xs0=Y[0,:]
    ys0=Y[1,:]
    # convert from meters to cells
    xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    # build an arbitrary map 
    indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
    MAP['map'][xis[indGood[0]],yis[indGood[0]]]=1
    for i in range(xis[indGood].shape[0]):
        MAP['map'][xis[indGood][i], yis[indGood][i]] += occupied   
    MAP['map'][MAP['map'] > occu_max] = occu_max # if the occupied accumulation is larger than 20
    MAP['map'][MAP['map'] < occu_min] = occu_min
    for i in range(N):
        x, y, theta = particles[i, :]   #.reshape(3, 1)
        x_range = np.arange(-0.2 + x, 0.2 + x + MAP['res'], MAP['res'])
        y_range = np.arange(-0.2 + y, 0.2 + y + MAP['res'], MAP['res'])
        correlation_particles = mapCorrelation(MAP['map'], x_im, y_im, Y, x_range, y_range)
        correlation_max = np.max(correlation_particles)
        correlation_max_pos = np.unravel_index(correlation_particles.argmax(), correlation_particles.shape)
        max_x_pos, max_y_pos = correlation_max_pos
        particles[i, 0] = -0.2 + x + max_x_pos * MAP['res']
        particles[i, 1] = -0.2 + y + max_y_pos * MAP['res']
        weights[i] *= np.exp(correlation_max / 100)
    weights /= np.sum(weights)  # normalize
    return particles,weights,MAP
'''    
def neff(weights):
    return 1. / np.sum(np.square(weights))

'''
def simple_resample(particles, weights):
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.  # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, random(N))
    # resample according to indexes
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)  # normalize
'''

def resample(particles, weights):
    j = 0; c = weights[0]
    N = len(particles)
    new_par = np.zeros([N,3])
    new_weight = np.ones([N,1])*(1/N)
    for k in range(N):
        u = np.random.uniform(0,1/N)
        b = u + (k-1)/N
        while b > c:
            j = j + 1; c = c + weights[j]
        new_par[k] = particles[j]
    return new_par, new_weight

def get_cell_robot(x,y,MAP):
    x_cell = np.ceil((x - MAP['xmin']) / 0.05 ).astype(np.int16) - 1
    y_cell = np.ceil((y - MAP['ymin']) / 0.05).astype(np.int16) - 1
    return x_cell,y_cell


def update(particles, weights, MAP,lidar_ranges,lidar_angles,Number):
    occupied = -1 #np.log(4)
    un_occupied = 1 #-np.log(4)
    occu_max = 20
    occu_min = -20
    best_particle = np.argmax(weights)
    x_robot, y_robot, theta_robot = particles[best_particle, :]
    x_im = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])  # x-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])  # y-positions of each pixel of the map

    '''
    # xy position in the sensor frame and convert position in the map frame here
    xs0 = lidar_ranges * np.cos(lidar_angles + theta_robot)
    ys0 = lidar_ranges * np.sin(lidar_angles + theta_robot)
    lidar_range_phy = np.stack((xs0, ys0))
    #Y = np.stack((xs0 + x_robot, ys0 + y_robot))
    # convert from meters to cells
    xl_end_map,yl_end_map = get_cell_robot(xs0+x_robot,ys0+y_robot,MAP)
    indGood = np.logical_and(np.logical_and(np.logical_and((xl_end_map > 1), (yl_end_map > 1)), (xl_end_map < MAP['sizex'])),
                             (yl_end_map < MAP['sizey']))
    '''
    
    
    xss = lidar_ranges * np.cos(lidar_angles + theta_robot)
    yss = lidar_ranges * np.sin(lidar_angles + theta_robot)
    YLC = np.stack((xss, yss))
    Y=np.zeros([2,1])
    YF=np.zeros([2,1])
    for i in range (np.size(lidar_ranges)):
        Yi=L2W(lidar_ranges[i], lidar_angles[i], x_robot, y_robot, theta_robot)
        xi=Yi[0]
        yi=Yi[1]
        xi=np.ceil((xi - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        yi=np.ceil((yi - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
        Y=np.hstack((Y,np.array([[xi],[yi]])))
        x_r_c=np.ceil((x_robot - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        y_r_c=np.ceil((y_robot - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
        YF=np.hstack((YF,bresenham2D(x_r_c, y_r_c, xi, yi)))
    Y=Y[:,1:]
    YF=YF[:,1:]
    #xs0=Y[0,:]
    #ys0=Y[1,:]
    xis=Y[0,:]
    yis=Y[1,:]
    xisf=YF[0,:]
    yisf=YF[1,:]
    # convert from meters to cells
    #xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    #yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    #xisf = np.ceil((xsf - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    #yisf = np.ceil((ysf - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    #x_r_c=np.ceil((x_robot - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    #y_r_c=np.ceil((y_robot - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    # build an arbitrary map 
    indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
    #indGoodf = np.logical_and(np.logical_and(np.logical_and((xisf > 1), (yisf > 1)), (xisf < MAP['sizex'])), (yisf < MAP['sizey']))

    
    '''
    
    for i in range(xl_end_map[indGood].shape[0]):
        MAP['map'][xl_end_map[indGood][i], yl_end_map[indGood][i]] += occupied   # If the end of lidar is in the map frame, Plus one...
    '''
    
    #x_r_c=np.ceil((x_robot - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    #y_r_c=np.ceil((y_robot - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
   
    for i in range(np.size(xis[indGood])):
        a=int(xis[indGood][i])
        b=int(yis[indGood][i])
        MAP['map'][a,b] = MAP['map'][a,b] + occupied
        #MAP['map'][xis[indGood][i], yis[indGood][i]] = 255
    for i in range(np.size(xisf)):
        if xisf[i]>1 and yisf[i]>1 and xisf[i]<MAP['sizex'] and yisf[i]<MAP['sizey']: 
            a=int(xisf[i])
            b=int(yisf[i])
            MAP['map'][a,b] = MAP['map'][a,b] + un_occupied 
        #MAP['map'][xisf[indGoodf][i], yisf[indGoodf][i]] = 100
    
    '''
    for i in range (xl_end_map[indGood].shape[0]):
        tempx=bresenham2D(x_r_c, y_r_c, xl_end_map[indGood][i], yl_end_map[indGood][i]).astype(np.int16)-1
        for j in range (len(tempx[0,:])-1):
            MAP['map'][tempx[0,j], tempx[1,j]]=MAP['map'][tempx[0,j], tempx[1,j]]+un_occupied
        MAP['map'][tempx[0,-1], tempx[1,-1]]=MAP['map'][tempx[0,-1], tempx[1,-1]]+occupied
    '''
    
    
    MAP['map'][MAP['map'] > occu_max] = occu_max 
    MAP['map'][MAP['map'] < occu_min] = occu_min

    for i in range(Number):
        x, y, theta = particles[i, :].reshape(3,1)   #.reshape(3, 1)
        x_range = np.arange(-0.5 + x, 0.5 + x + MAP['res'], MAP['res'])
        y_range = np.arange(-0.5 + y, 0.5 + y + MAP['res'], MAP['res'])
        #correlation_particles = mapCorrelation(MAP['map'], x_im, y_im, lidar_range_phy, x_range, y_range)
        correlation_particles = mapCorrelation(MAP['map'], x_im, y_im, YLC, x_range, y_range)
        correlation_max = np.max(correlation_particles)
        correlation_max_pos = np.unravel_index(correlation_particles.argmax(), correlation_particles.shape)
        max_x_pos, max_y_pos = correlation_max_pos
        particles[i, 0] = -0.5 + x + max_x_pos * MAP['res']
        particles[i, 1] = -0.5 + y + max_y_pos * MAP['res']
        weights[i] *= np.exp(correlation_max / 100)
    weights /= np.sum(weights)  # normalize
    '''
    a0=np.ceil((x_robot - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    b0=np.ceil((y_robot - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    bp = np.argmax(weights)
    aa, bb, theta_bp = particles[bp, :]
    a1=np.ceil((aa - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    b1=np.ceil((bb - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    plt.plot([a0,b0], [a1,b1], color='r', linewidth=2)
    '''
    '''
    a0=x_robot+601
    b0=y_robot+601
    bp = np.argmax(weights)
    aa, bb, theta_bp = particles[bp, :]
    a1=aa+601
    b1=bb+601
    plt.plot([a0,b0], [a1,b1], color='r', linewidth=1)
    '''
    return particles,weights,MAP,x_robot, y_robot
    

N=6
particles=create_particles([-0.5,0.5], [-0.5,0.5],0,N)
weights = np.ones([N, 1]) / N

# init MAP
MAP = {}
MAP['res']   = 0.05 #meters
MAP['xmin']  = -30  #meters
MAP['ymin']  = -30
MAP['xmax']  =  30
MAP['ymax']  =  30 
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8)
#MAP['map'] = 20*np.ones((MAP['sizex'],MAP['sizey']),dtype=np.int8) 

PC=601*np.ones([2,1])
for i in range (np.size(encoder_stamps)-5):
    t1=encoder_stamps[i]
    t2=encoder_stamps[i+1]
    ranges=lidar_ranges[:,i]
    angles = np.arange(-135, 135.25, 0.25) * np.pi / 180.0
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    dt=t2-t1
    ec=encoder_counts[:,i]
    d=dist(ec)
    particles = predict(particles, dt, d, vyaw[i], theta_w[i,0])  # (X，Y) Unit:meters
    particles,weights, MAP, xc, yc = update(particles, weights,MAP,ranges,angles,N)
    xc=xc/0.05+601
    yc=yc/0.05+601
    #plt.plot([PC[0,i], PC[1,i]], [xc,yc], color='r')
    PC=np.hstack((PC,np.array([[xc], [yc]])))
    if neff(weights) < N/2:
        #simple_resample(particles, weights)
        particles, weights=resample(particles, weights)
    print(i)


np.save('position21.npy', PC)
np.savetxt('position21.txt', PC)
plt.imshow(MAP['map'], cmap='gray')
aa=np.zeros([1201,1201])
for i in range (1201):
    for j in range (1201):
        if MAP['map'][i,j]>15:
            aa[i,j]=255
cv2.imwrite('aa21.jpg',aa)
plt.gca().invert_yaxis()
plt.colorbar()
#plt.plot(PC[1,:],PC[0,:],color='r')
plt.show

'''
for i in range (0,1999):
    plt.plot([PC[0,i], PC[1,i]], [PC[0,1+i], PC[1,i+1]], color='r')0.2
'''

print(1)