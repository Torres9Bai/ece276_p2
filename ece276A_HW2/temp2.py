# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 08:19:52 2019

@author: User
"""

import numpy as np
import math
from matplotlib import pyplot as plt
import cv2
import skimage.io as io
import time

def tic():
  return time.time()
def toc(tstart, name="Operation"):
  print('%s took: %s sec.\n' % (name,(time.time() - tstart)))


if __name__ == '__main__':
  dataset = 20
  
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
    w=w[0:2,:]
    return w

L2C=np.array([[1,0,13.323],
              [0,1,0],
              [0,0,1]])

Lidar1=lidar_ranges[:,0]

ww=np.zeros([2,1])
for i in range (np.size(Lidar1)):
    if Lidar1[i]<lidar_range_max and Lidar1[i]>lidar_range_min:
        theta=lidar_angle_min+i*lidar_angle_increment
        xi=100*Lidar1[i]*math.cos(theta)
        yi=100*Lidar1[i]*math.sin(theta)
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

xL0=13.323
yL0=0.

time_start = tic()
wo=np.zeros([2,1])
for i in range (ww.shape[1]):
    wo0=bresenham2D(xL0, yL0, ww[0,i], ww[1,i])
    wo=np.hstack((wo,wo0))
toc(time_start)
wo=wo[:,1:]

#wmap0=np.zeros([81,81])
wmap0=255*np.ones([8001,8001])

'''
for i in range (wo.shape[1]):
    a0=int(wo[0,i])
    b0=int(wo[1,i])
    a=a0+41
    b=41-b0
    wmap0[b,a]=255

io.imshow(wmap0)
'''
for i in range (ww.shape[1]):
    a0=int(ww[0,i])
    b0=int(ww[1,i])
    a=a0+4001
    b=4001-b0
    wmap0[b,a]=0

io.imshow(wmap0)