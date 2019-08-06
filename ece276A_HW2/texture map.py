# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 12:09:18 2019

@author: User
"""

import numpy as np
import math
from matplotlib import pyplot as plt
import cv2
import skimage.io as io
import time
from numpy.random import uniform,randn,random
#from compiler.ast import flatten


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
    

disp=io.ImageCollection(r'D:\\BYS\\UCSD\\ece276A\\ECE276A_HW2\\ECE276A_HW2\\dataRGBD\\Disparity21\\*.png')
rgb=io.ImageCollection(r'D:\\BYS\\UCSD\\ece276A\\ECE276A_HW2\\ECE276A_HW2\\dataRGBD\\RGB21\\*.png')

#io.imshow(rgb[10])

def Euler_angle(yaw, pich, roll):
    R_yaw=np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    R_pich=np.array([[math.cos(pich), 0, math.sin(pich)], [0, 1, 0], [-math.sin(pich), 0, math.cos(pich)]])
    R_roll=np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    R=np.dot(np.dot(R_yaw, R_pich), R_roll)
    return R

def xj2w(xc,yc,yaw):
    Sb=np.array([[0.18], [0.005], [0.38], [1.]])
    T=np.array([[math.cos(yaw), -math.sin(yaw), 0, xc], [math.sin(yaw), math.cos(yaw), 0, yc], [0, 0, 1, 0.127], [0, 0, 0, 1]])
    Sw=np.dot(T, Sb)
    Sw=Sw[0:3, 0]
    return Sw

'''
map20=cv2.imread('aa20.jpg')
B, G, R = cv2.split(map20)
map20 = cv2.merge([R, G, B])
r_arr = np.array(R).reshape(1201, 1201)
g_arr = np.array(G).reshape(1201, 1201)
b_arr = np.array(B).reshape(1201, 1201)
'''
r_arr = np.zeros([1201, 1201])
g_arr = np.zeros([1201, 1201])
b_arr = np.zeros([1201, 1201])


PC=np.load('position21.npy')

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

'''
j=0
k=0
for i in range (1):
    t0=rgb_stamps[i]
    while (imu_stamps[j]<t0):
        j=j+1
    while (disp_stamps[k]<t0):
        k=k+1
    Kante=np.zeros([2,1])
    depth=np.zeros([1,1])
    dispi=disp[k-1]
    for m in range (480):
        for n in range (640):
            Kanteu,Kantev,depthi=CR7(m,n,dispi[m,n])
            if Kanteu<480 and Kanteu>1 and Kantev<640 and Kantev>1:
                tempK=np.array([[Kanteu], [Kantev]])
                Kante=np.hstack((Kante, tempK))
                depth=np.hstack((depth, depthi.reshape(1,1)))
    Kante=Kante[:,1:]
    depth=depth[0,1:]
'''
j=130
k=0
for i in range (np.size(rgb_stamps)-1):
    t0=rgb_stamps[i]
    while (encoder_stamps[j]<t0):
        j=j+1
    #t1=rgb_stamps[j]
    while (disp_stamps[k]<t0):
        k=k+1
    mn = np.zeros((480,640,2))
    for m in range (480):
        for n in range (640):
            mn[m,n,0]=m
            mn[m,n,1]=n   
        #print(m)
    rowi=mn[:,:,0].reshape(-1,1).T
    rowj=mn[:,:,1].reshape(-1,1).T
    d=disp[k-1].flatten().reshape(1,480*640)
    dd=-0.00304*d+3.31
    depth=1.03/dd
    ui=(rowi*526.37+dd*(-4.5*1750.46)+19276.0)/585.051
    vi=(rowj*526.37+16662.0)/585.051
    indGood = np.logical_and(np.logical_and(np.logical_and((ui > 1), (vi > 1)), (ui < 480)), (vi < 640))
    Kante=np.vstack((ui[indGood], vi[indGood], np.ones([1,np.size(ui[indGood])])))
    #Po=np.dot(depthi*np.dot(np.linalg.inv(K), Kante)
    IKA = np.linalg.inv(K) @ Kante
    IKA *= depth[indGood][:]
    xc=(PC[0,j]-601)*0.05
    yc=(PC[1,j]-601)*0.05
    Rwc=Euler_angle(0.021+theta_w[j,0], 0.36, 0)
    Pwc=xj2w(xc, yc, theta_w[j,0])
    Tow1=np.dot(Roc, Rwc.T)
    Tow2=np.dot(np.dot(-Roc, Rwc.T), Pwc).reshape(3,1)
    Tow=np.hstack((Tow1, Tow2))
    Tow=np.vstack((Tow, np.array([0, 0, 0, 1])))
    IKA=np.vstack((IKA, np.ones([1,IKA.shape[1]])))
    Pw=np.linalg.inv(Tow) @ IKA
    goodp=Pw[:,np.logical_and(Pw[2]<0.4,Pw[2]>-0.2)]
    Kepa=Tow @ goodp
    Kepa[0,:]=Kepa[0,:]/Kepa[2,:]
    Kepa[1,:]=Kepa[1,:]/Kepa[2,:]
    Kepa[2,:]=Kepa[2,:]/Kepa[2,:]
    Luiz=np.dot(K, Kepa[0:3, :])
    RR, GG, BB=cv2.split(rgb[i])
    ri=np.array(RR)
    gi=np.array(GG)
    bi=np.array(BB)
    for q in range (Luiz.shape[1]):
        a=int(goodp[0,q]/0.05+601)
        b=int(goodp[1,q]/0.05+601)
        aa=int(Luiz[0,q])
        bb=int(Luiz[1,q])
        r_arr[a,b]=ri[aa,bb]
        g_arr[a,b]=gi[aa,bb]
        b_arr[a,b]=bi[aa,bb]
        #print(q)
    print(i)

result=cv2.merge([b_arr, g_arr, r_arr])
#result=cv2.merge([r_arr, g_arr, b_arr])
#cv2.imshow(result)
cv2.imwrite('tm21aa.jpg',result) 