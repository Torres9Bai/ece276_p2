# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 03:12:34 2019

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
'''
map20=cv2.imread('temp20.png')
aa=np.zeros([1201,1201])
for i in range (1201):
    for j in range (1201):
        if map20[i,j]>5:
            aa[i,j]=255
            
cv2.imwrite('aa20.jpg',aa)
'''

disp=io.ImageCollection(r'D:\\BYS\\UCSD\\ece276A\\ECE276A_HW2\\ECE276A_HW2\\dataRGBD\\Disparity20\\*.png')
rgb=io.ImageCollection(r'D:\\BYS\\UCSD\\ece276A\\ECE276A_HW2\\ECE276A_HW2\\dataRGBD\\RGB20\\*.png')

#io.imshow(rgb[10])

def Euler_angle(yaw, pich, roll):
    R_yaw=np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    R_pich=np.array([[math.cos(pich), 0, math.sin(pich)], [0, 1, 0], [-math.sin(pich), 0, math.cos(pich)]])
    R_roll=np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    R=np.dot(np.dot(R_yaw, R_pich), R_roll)
    return R

def xj2w(xc,yc,yaw):
    Sb=np.array([[0.18], [0.005], [0.36], [1.]])
    T=np.array([[math.cos(yaw), -math.sin(yaw), 0, xc], [math.sin(yaw), math.cos(yaw), 0, yc], [0, 0, 1, 0.127], [0, 0, 0, 1]])
    Sw=np.dot(T, Sb)
    Sw=Sw[0:3, 0]
    return Sw


map20=cv2.imread('aa20.jpg')
B, G, R = cv2.split(map20)
map20 = cv2.merge([R, G, B])
r_arr = np.array(R).reshape(1201, 1201)
g_arr = np.array(G).reshape(1201, 1201)
b_arr = np.array(B).reshape(1201, 1201)


PC=np.load('position20.npy')

Roc=np.array([[0,-1,0], [0, 0, -1], [1, 0, 0]])
K=np.array([[585.05108211,0,242.94140713], [0, 585.05108211, 315.83800193], [0, 0, 1]])


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

def CR7(i, j, d):
    #dd=−0.00304∗d+3.31
    dd=-0.00304*d+3.31
    depth=1.03/dd
    rgbi=(i*526.37+dd*(-4.5*1750.46)+19276.0)/585.051
    rgbj=(j*526.37+16662.0)/585.051
    return rgbi, rgbj, depth


j=0
k=0
groud=np.zeros([2,1])
#np.size(rgb_stamps)-1
for i in range (1):
    t0=rgb_stamps[i]
    while (imu_stamps[j]<t0):
        j=j+1
    while (disp_stamps[k]<t0):
        k=k+1
    xc=(PC[0,j-1]-601)*0.05
    yc=(PC[1,j-1]-601)*0.05
    Rwc=Euler_angle(0.021+theta_w[j-1,0], 0.36, 0)
    Pwc=xj2w(xc, yc, theta_w[j-1,0])
    Tow1=np.dot(Roc, Rwc.T)
    Tow2=np.dot(np.dot(-Roc, Rwc.T), Pwc).reshape(3,1)
    Tow=np.hstack((Tow1, Tow2))
    Tow=np.vstack((Tow, np.array([0, 0, 0, 1])))
    dispi=disp[k-1]
    #B, G, R = cv2.split(dispi)
    R = cv2.split(dispi)
    d_value=np.array(R).reshape(480, 640)
    for m in range (480):
        for n in range (640):
            #rgbm, rgbn, depthi=CR7(m, n, d_value[m,n])
            rgbm, rgbn, depthi=CR7(n, m, d_value[m,n])
            if rgbm<640 and rgbn<480:
                Po=depthi*np.dot(np.linalg.inv(K), np.array([[rgbm], [rgbn], [1]]))
                Po=np.vstack((Po, np.ones((1,1))))
                Pw=np.dot(np.linalg.inv(Tow), Po)
                if np.abs(Pw[2,0])<0.1:
                    goodP=np.vstack((Pw[0,0], Pw[1,0]))
                    groud=np.hstack((groud, goodP))
                    a=int(Pw[0,0]/0.05+601)
                    b=int(Pw[1,0]/0.05+601)
                    RR, GG, BB=cv2.split(rgb[i])
                    ri=np.array(RR).reshape(640, 480)
                    gi=np.array(GG).reshape(640, 480)
                    bi=np.array(BB).reshape(640, 480)
                    r_arr[a,b]=ri[int(rgbm), int(rgbn)]
                    g_arr[a,b]=gi[int(rgbm), int(rgbn)]
                    b_arr[a,b]=bi[int(rgbm), int(rgbn)]
    print(i)
    
                
    
result=cv2.merge([r_arr, g_arr, b_arr])
io.imshow(result)
    
    