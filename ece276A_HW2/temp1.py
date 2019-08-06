# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 07:27:38 2019

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

'''
L2C=np.array([[1,0,0.13323],
              [0,1,0],
              [0,0,1]])

    
    
'''
L2C=np.array([[1,0,1.3323],
              [0,1,0],
              [0,0,1]])


Lidar1=lidar_ranges[:,0]

ww=np.zeros([2,1])
for i in range (np.size(Lidar1)):
    if Lidar1[i]<lidar_range_max and Lidar1[i]>lidar_range_min:
        theta=lidar_angle_min+i*lidar_angle_increment
        xi=10*Lidar1[i]*math.cos(theta)
        yi=10*Lidar1[i]*math.sin(theta)
        p1=np.array([[xi],[yi],[1]])
        p2=np.dot(L2C,p1)
        pw=p2[0:2,:]
        ww=np.hstack((ww,pw))

ww=ww[:,1:]

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

xL0=1.3323
yL0=0.

wo=np.zeros([2,1])#扫描之后发现是空的点
wo0_La=np.zeros([3,1])#带label的所有扫描点
wob=np.zeros([2,1])#边界点
for i in range (ww.shape[1]):
    wo0=bresenham2D(xL0, yL0, ww[0,i], ww[1,i])
    wobi=np.array([[wo0[0,-1]],[wo0[1,-1]]])
    wob=np.hstack((wob,wobi))
    wo0f=wo0[:,0:(wo0.shape[1]-1)]
    wo=np.hstack((wo,wo0f))
    wo0_L=np.vstack((wo0,1/4*np.ones([1,wo0.shape[1]])))
    wo0_L[-1,-1]=4
    wo0_La=np.hstack((wo0_La,wo0_L))

wo=wo[:,1:]
wo0_La=wo0_La[:,1:]
wob=wob[:,1:]

wmap0=np.zeros([801,801])
#wmap0=255*np.ones([801,801])


for i in range (wo.shape[1]):
    a0=int(wo[0,i])
    b0=int(wo[1,i])
    a=a0+401
    b=401+b0
    wmap0[b,a]=255
    


for i in range (wob.shape[1]):
    a0=int(wob[0,i])
    b0=int(wob[1,i])
    a=a0+401
    b=401+b0
    wmap0[b,a]=150

'''
#io.imshow(wmap0)
plt.imshow(wmap0)
plt.gca().invert_yaxis()
plt.show
'''

def dist (encoder):
    dL=(encoder[1]+encoder[3])/2*0.0022
    dR=(encoder[0]+encoder[2])/2*0.0022
    dist=(dL+dR)/2
    return dist


'''
j=0
cpw=np.zeros([1,2])
theta_w=np.zeros([1,1])
vw=np.zeros([1,2])
dd=np.zeros([1,1])
theta_t=0
for i in range (np.size(encoder_stamps)-1):
    t1=encoder_stamps[i]
    t2=encoder_stamps[i+1]
    #theta_t=0
    while (imu_stamps[j]<t2):
        if j>(np.size(imu_stamps)-2):
            break
        theta_t=theta_t+imu_lpvyaw[j]*(imu_stamps[j+1]-imu_stamps[j])
        j=j+1
    ec=encoder_counts[:,i]
    d=dist(ec)
    dd=np.vstack((dd,d))
    vx=d*math.cos(theta_t)/(t2-t1)
    vy=d*math.sin(theta_t)/(t2-t1)
    dt=t2-t1
    v=d/dt
    vyaw=theta_t/dt
    if vyaw!=0:
        xcw=cpw[i,0]+v*dt*math.sin(vyaw*dt/2)/(vyaw*dt/2)*math.cos(theta_w[i]+vyaw*dt/2)
        ycw=cpw[i,1]+v*dt*math.sin(vyaw*dt/2)/(vyaw*dt/2)*math.sin(theta_w[i]+vyaw*dt/2)
    else:
        xcw=cpw[i,0]+v*dt*1*math.cos(theta_w[i]+vyaw*dt/2)
        ycw=cpw[i,1]+v*dt*1*math.sin(theta_w[i]+vyaw*dt/2)
    vw=np.vstack((vw,np.array([vx,vy])))
    cpw=np.vstack((cpw,np.array([xcw,ycw])))
    theta_w=np.vstack((theta_w,theta_t))
'''    

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
    

def create_particles(x_range, y_range, theta, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = theta
    return particles


x_range=np.array([0,801])
y_range=np.array([0,801])    
ptc=create_particles(x_range, y_range, 0, 100)


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
        xcw=x0+v*dt*math.sin(vyaw*dt/2)/(vyaw*dt/2)*math.cos(theta0+vyaw*dt/2)+np.random.normal(0,0.02)
        ycw=y0+v*dt*math.sin(vyaw*dt/2)/(vyaw*dt/2)*math.sin(theta0+vyaw*dt/2)+np.random.normal(0,0.02)
    else:
        xcw=x0+v*dt*1*math.cos(theta0+vyaw*dt/2)+np.random.normal(0,0.02)
        ycw=y0+v*dt*1*math.sin(theta0+vyaw*dt/2)+np.random.normal(0,0.02)
    position=np.array([xcw, ycw])
    return position

cpw=np.zeros([1,2])
for i in range (np.size(encoder_stamps)-1):
    t1=encoder_stamps[i]
    t2=encoder_stamps[i+1]
    dt=t2-t1
    ec=encoder_counts[:,i]
    d=dist(ec)
    position=ptc_motion(cpw[i,0], cpw[i,1], theta_w[i,0], d, vyaw[i,0], dt)
    cpw=np.vstack((cpw, position))




def pgrid(x,y):
    xx=bresenham2D(x,y,x-4,y)
    xx=np.hstack((xx, bresenham2D(x,y,x+4,y)[:,1:]))
    xx=xx[0]
    yy=bresenham2D(x,y,x,y-4)
    yy=np.hstack((yy, bresenham2D(x,y,x,y+4)[:,1:]))
    yy=yy[1]
    #p=np.vstack((xx,yy))
    p=np.zeros([2,1])
    for i in range (0,9):
        for j in range (0,9):
            px=xx[i]
            py=yy[j]
            pi=np.vstack((px, py))
            p=np.hstack((p,pi))
    p=p[:,1:]
    return p


'''

Lidar2=lidar_ranges[:,1]

ww1=np.zeros([2,1])
for i in range (np.size(Lidar1)):
    if Lidar1[i]<lidar_range_max and Lidar1[i]>lidar_range_min:
        theta=lidar_angle_min+i*lidar_angle_increment
        xi=10*Lidar1[i]*math.cos(theta)
        yi=10*Lidar1[i]*math.sin(theta)
        p1=np.array([[xi],[yi],[1]])
        p2=np.dot(L2C,p1)
        pw=p2[0:2,:]
        ww1=np.hstack((ww1,pw))

ww1=ww1[:,1:]
testptc1=create_particles([391,411],[391,411],0,100)

testwo=np.zeros([2,1])#扫描之后发现是空的点
testwo0_La=np.zeros([3,1])#带label的所有扫描点
testwob=np.zeros([2,1])#边界点
for i in range (ww1.shape[1]):
    testwo0=bresenham2D(xL0, yL0, ww1[0,i], ww1[1,i])
    testwobi=np.array([[testwo0[0,-1]+401],[testwo0[1,-1]+401]])
    testwob=np.hstack((testwob,testwobi))
    testwo0f=testwo0[:,0:(testwo0.shape[1]-1)]
    testwo=np.hstack((testwo,testwo0f))
    testwo0_L=np.vstack((testwo0,1/4*np.ones([1,testwo0.shape[1]])))
    testwo0_L[-1,-1]=4
    testwo0_La=np.hstack((testwo0_La,testwo0_L))

testwo=testwo[:,1:]
testwo0_La=testwo0_La[:,1:]
testwob=testwob[:,1:]

testdt=encoder_stamps[1]-encoder_stamps[0]
testd=dist(encoder_counts[:,1])/100

testpp=np.zeros([1,2])
for i in range (0,100):
    x0=testptc1[i,0]
    y0=testptc1[i,1]
    testppi=ptc_motion(x0,y0,theta_w[0],testd,vyaw[1,0],testdt)
    testpp=np.vstack((testpp, testppi))

testpp=testpp[1:,:]
    
testc1=mapCorrelation(wmap0, x_range, y_range, testwob, testpp[:,0], testpp[:,1])
'''

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
    


def predict(particles, dt, d, vyaw):
    #particles[:,2]=particles[:,2]+vyaw*dt
    for i in range (0,particles.shape[0]):
        xi, yi=ptc_motion(particles[i,0], particles[i,1], particles[i,2],d,vyaw,dt)
        particles[i,0]=xi
        particles[i,1]=yi
    return particles

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
    MAP['map'][MAP['map'] > occu_max] = occu_max # if the occupied accumulation is larger than 20
    MAP['map'][MAP['map'] < occu_min] = occu_min
    for i in range(N):
        x, y, theta = particles[i, :]   #.reshape(3, 1)
        x_range = np.arange(-0.4 + x, 0.4 + x + MAP['res'], MAP['res'])
        y_range = np.arange(-0.4 + y, 0.4 + y + MAP['res'], MAP['res'])
        correlation_particles = mapCorrelation(MAP['map'], x_im, y_im, Y, x_range, y_range)
        correlation_max = np.max(correlation_particles)
        correlation_max_pos = np.unravel_index(correlation_particles.argmax(), correlation_particles.shape)
        max_x_pos, max_y_pos = correlation_max_pos
        particles[i, 0] = -0.4 + x + max_x_pos * MAP['res']
        particles[i, 1] = -0.4 + y + max_y_pos * MAP['res']
        weights[i] *= np.exp(correlation_max / 100)
    weights /= np.sum(weights)  # normalize
    return particles,weights,MAP
    
def neff(weights):
    return 1. / np.sum(np.square(weights))


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
N=6
particles=create_particles([-0.4,0.4], [-0.4,0.4],0,N)
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
MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8

for i in range (np.size(encoder_stamps)-1):
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
    particles = predict(particles, dt, d, vyaw[i])  # (X，Y) Unit:meters
    particles,weights, MAP = update(particles, weights,MAP,ranges,angles,N)
    if neff(weights) < N/2:
        simple_resample(particles, weights)
    print(i)

plt.imshow(MAP['map'])
plt.gca().invert_yaxis()
plt.show

print(1)

'''
    
'''
plt.plot(imu_vyaw)
plt.plot(imu_lpvyaw)   
plt.show
'''
cpp=np.zeros([2,1])
for i in range (cpw.shape[0]-1):
    xi=cpw[i,0]
    yi=cpw[i,1]
    pi=pgrid(xi,yi)
    cpp=np.hstack((cpp,pi))

pp0=np.zeros([81,81])
#wmap0=255*np.ones([801,801])


for i in range (cpp.shape[1]):
    a0=int(cpp[0,i])
    b0=int(cpp[1,i])
    a=a0+41
    b=41+b0
    pp0[b,a]=255
    
plt.imshow(pp0)
plt.gca().invert_yaxis()
plt.show
