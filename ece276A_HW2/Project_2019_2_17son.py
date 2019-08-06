import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import cv2
#from load_data import load_data
import time
from scipy import signal
from numpy.random import uniform,randn,random

# Define the time calculating function
def tic():
    return time.time()

def toc(tstart, name="Operation"):
    print('%s took: %s sec.\n' % (name, (time.time() - tstart)))


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


def get_cell_lidar(Xb,Yb,ranges,angles,MAP): #获取lidar结束点的cell坐标
    x = Xb + ranges * np.cos(angles)
    y = Yb + ranges * np.sin(angles)
    x_cell = np.ceil((x - MAP['xmin']) / 0.05 ).astype(np.int16) - 1
    y_cell = np.ceil((y - MAP['ymin']) / 0.05 ).astype(np.int16) - 1
    return np.vstack((x_cell,y_cell))


def get_cell_robot(x,y,MAP):
    x_cell = np.ceil((x - MAP['xmin']) / 0.05 ).astype(np.int16) - 1
    y_cell = np.ceil((y - MAP['ymin']) / 0.05).astype(np.int16) - 1
    return x_cell,y_cell

def get_angular_velocity(angular_velocity_lasttime, time):#就用最新的IMU-Time即可。
    imu_tmp = np.where(imu_stamps <= encoder_stamps[time+1])
    imu_tmp_stamps = imu_stamps[imu_tmp]
    x = np.where(imu_tmp_stamps >= encoder_stamps[time])
    imu_tmp = yaw_data[x]
    #imu_tmp = yaw_data[2,x]
    if len(imu_tmp[0,:])==0:
        omega = angular_velocity_lasttime
    else:
        omega =  (np.sum(imu_tmp)/len(imu_tmp[0,:]))  #修改了这里为[2,:]
        angular_velocity_lasttime  =omega
    return omega, angular_velocity_lasttime



def get_theta(angular_velocity, particles, time_interval):
    return angular_velocity * time_interval


def get_velocity(time_encoder):
    Vr = (encoder_counts[0,time_encoder]+encoder_counts[2,time_encoder])/2*0.0022
    Vl = (encoder_counts[1,time_encoder]+encoder_counts[3,time_encoder])/2*0.0022
    Velocity = (Vr+Vl)/2
    return Velocity


def get_updated_position(velocity, angular_velocity,time_interval,particles,Number):
    particles = calibrate_theta(particles, Number)
    velocity=velocity/time_interval
    particles[:,0] = particles[:,0] + velocity * time_interval*np.sin(angular_velocity*time_interval/2)/(angular_velocity*time_interval\
    /2)*np.cos(particles[:,2] + angular_velocity*time_interval/2)

    particles[:, 1] = particles[:,1] + velocity * time_interval * np.sin(angular_velocity * time_interval /2)/(angular_velocity * \
    time_interval /2) * np.sin(particles[:,2] + angular_velocity * time_interval/2)
    return particles


def set_particles(x_range, y_range, theta_range, Number):
    particles = np.empty((Number, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=Number)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=Number)
    particles[:, 2] = uniform(theta_range[0], theta_range[1], size=Number)
    return particles


def calibrate_theta(particles,Number):
    for i in range(Number):
        if particles[i,2] > np.pi:
            particles[i,2] %= 2*np.pi
            particles[i,2] = -(2*np.pi-particles[i,2])
        elif particles[i,2] < -np.pi:
            particles[i, 2] = (-particles[i, 2])%(2 * np.pi)
            particles[i, 2] = 2 * np.pi - particles[i, 2]
        else:
            pass
    return particles


def predict(time, particles, time_interval,Number,angular_velocity):
    # global angular_velocity_lasttime
    # average_angular_velocity, angular_velocity_lasttime = get_angular_velocity(angular_velocity_lasttime, time)
    #particles[:, 2] += average_angular_velocity * time_interval
    particles[:, 2] += time_interval * angular_velocity
    for i in range(Number):
        if particles[i,2] > np.pi:
            particles[i,2] %= 2*np.pi
            particles[i,2] = -(2*np.pi-particles[i,2])
        elif particles[i,2] < -np.pi:
            particles[i, 2] = (-particles[i, 2])%(2 * np.pi)
            particles[i, 2] = 2 * np.pi - particles[i, 2]
        else:
            pass
    average_velocity = get_velocity(time)
    particles = get_updated_position(average_velocity,angular_velocity,time_interval,particles, Number)
    #x_cell, y_cell = get_cell_robot(x_position,y_position)
    return particles


def update(particles, weights, MAP,lidar_ranges,lidar_angles,Number):
    occupied = 1
    un_occupied = -1
    occu_max = 20
    occu_min = -20
    best_particle = np.argmax(weights)    #这样为何可以得到最大的weight，怎么理解呢？
    x_robot, y_robot, theta_robot = particles[best_particle, :]
    x_im = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])  # x-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])  # y-positions of each pixel of the map

    # xy position in the sensor frame and convert position in the map frame here
    xs0 = lidar_ranges * np.cos(lidar_angles + theta_robot)
    ys0 = lidar_ranges * np.sin(lidar_angles + theta_robot)
    lidar_range_phy = np.stack((xs0, ys0))
    Y = np.stack((xs0 + x_robot, ys0 + y_robot))
    # convert from meters to cells
    xl_end_map,yl_end_map = get_cell_robot(xs0+x_robot,ys0+y_robot,MAP)
    indGood = np.logical_and(np.logical_and(np.logical_and((xl_end_map > 1), (yl_end_map > 1)), (xl_end_map < MAP['sizex'])),
                             (yl_end_map < MAP['sizey']))
    x_r_c=np.ceil((x_robot - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    y_r_c=np.ceil((y_robot - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    for i in range (xl_end_map[indGood].shape[0]):
        tempx=bresenham2D(x_r_c, y_r_c, xl_end_map[indGood][i], yl_end_map[indGood][i]).astype(np.int16)-1
        for j in range (len(tempx[0,:])-1):
            MAP['map'][tempx[0,j], tempx[1,j]]=MAP['map'][tempx[0,j], tempx[1,j]]+un_occupied
        MAP['map'][tempx[0,-1], tempx[1,-1]]=MAP['map'][tempx[0,-1], tempx[1,-1]]+occupied
    # for i in range(xis[indGood].shape[0]):
    # frees = bresenham2D(robot_x_cell, robot_y_cell, xis[indGood][i], yis[indGood][i])
    # frees = np.array(frees)
    # MAP['map'][frees] += lo_free
    # for j in range(frees.shape[1]):
    # MAP['map'][frees[0, j], frees[1, j]] += lo_free
    MAP['map'][MAP['map'] > occu_max] = occu_max # if the occupied accumulation is larger than 20
    MAP['map'][MAP['map'] < occu_min] = occu_min

    for i in range(Number):
        x, y, theta = particles[i, :].reshape(3,1)   #.reshape(3, 1)
        x_range = np.arange(-0.2 + x, 0.2 + x + MAP['res'], MAP['res'])
        y_range = np.arange(-0.2 + y, 0.2 + y + MAP['res'], MAP['res'])
        correlation_particles = mapCorrelation(MAP['map'], x_im, y_im, lidar_range_phy, x_range, y_range)
        correlation_max = np.max(correlation_particles)
        correlation_max_pos = np.unravel_index(correlation_particles.argmax(), correlation_particles.shape)
        max_x_pos, max_y_pos = correlation_max_pos
        particles[i, 0] = -0.2 + x + max_x_pos * MAP['res']
        particles[i, 1] = -0.2 + y + max_y_pos * MAP['res']
        weights[i] *= np.exp(correlation_max / 100)
    weights /= np.sum(weights)  # normalize
    return particles,weights,MAP


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
    iy = np.int16(np.round((y1-ymin)/yresolution)) #求出Y方向的点对应的grid数
    for jx in range(0,nxs):
      x1 = vp[0,:] + xs[jx] # 1 x 1076
      ix = np.int16(np.round((x1-xmin)/xresolution)) #求出X方向的点对应的数
      valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)),np.logical_and((ix >=0), (ix < nx)))
      cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
  return cpr


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
    return particles,weights


def showMap(MAP):
    fig = plt.figure()
    plt.imshow(MAP['map'].T[::-1], cmap='Greys',extent=(MAP['xmin'],MAP['xmax'],MAP['ymin'],MAP['ymax']))
    plt.show()

if __name__ == '__main__':
    """这里是进行数据的读取"""
    dataset = 20
    with np.load("Encoders%d.npz" % dataset) as data:
        encoder_counts = data["counts"]  # 4 x n encoder counts
        encoder_stamps = data["time_stamps"]  # encoder time stamps
    
    with np.load("Hokuyo%d.npz" % dataset) as data:
        lidar_angle_min = data["angle_min"]  # start angle of the scan [rad]
        lidar_angle_max = data["angle_max"]  # end angle of the scan [rad]
        lidar_angle_increment = data["angle_increment"]  # angular distance between measurements [rad]
        lidar_range_min = data["range_min"]  # minimum range value [m]
        lidar_range_max = data["range_max"]  # maximum range value [m]
        lidar_ranges = data["ranges"]  # range data [m] (Note: values < range_min or > range_max should be discarded)
        lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
    
    with np.load("Imu%d.npz" % dataset) as data:
        imu_angular_velocity = data["angular_velocity"]  # angular velocity in rad/sec
        imu_linear_acceleration = data["linear_acceleration"]  # Accelerations in gs (gravity acceleration scaling)
        imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
    
    with np.load("Kinect%d.npz" % dataset) as data:
        disp_stamps = data["disparity_time_stamps"]  # acquisition times of the disparity images
        rgb_stamps = data["rgb_time_stamps"]  # acquisition times of the rgb images
    
    '''
    encoder_counts, encoder_stamps, lidar_angle_min, lidar_angle_max, lidar_angle_increment, lidar_range_min, \
    lidar_range_max, lidar_ranges, lidar_stamps, imu_angular_velocity, imu_linear_acceleration, imu_stamps, \
            disp_stamps, rgb_stamps = load_data()
    '''

    
    encoder_time = 0
    time_lidar = 0
    time_encoder = 0
    count = 0
    x_robot_world = 0
    y_robot_world = 0
    Number = 5
    # particles = np.zeros((Number,3))
    # particles[:][3] = 1
    # particles[:][1] = 2
    particles = set_particles((-0.2, 0.2), (-0.2, 0.2), (0, 0), 5) # setup Particles
    weights = np.ones((Number, 1)) / Number
    angular_velocity_lasttime = 0

    begin_stamps = np.array([encoder_stamps[0], imu_stamps[0], lidar_stamps[0], rgb_stamps[0]])
    end_stamps = np.array([encoder_stamps[-1], imu_stamps[-1], lidar_stamps[-1], rgb_stamps[-1]])
    begin_time = np.max(begin_stamps).astype(np.int) + 1
    end_time = np.max(end_stamps).astype(np.int) - 1
    begin_encoder_index = np.min(np.where(encoder_stamps.astype(np.int) == begin_time))
    end_encoder_index = np.min(np.where(encoder_stamps.astype(np.int) == end_time))

    #test_mapCorrelation(lidar_ranges,lidar_range_max,lidar_range_min,lidar_angle_max,lidar_angle_min,0)
    MAP = {}
    MAP['res'] = 0.05  # meters
    MAP['xmin'] = -15  # meters
    MAP['ymin'] = -15
    MAP['xmax'] = 30
    MAP['ymax'] = 30
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.int8)  # MAP的对象里面创建了MAP这个变量。DATA TYPE: char or int8

    b, a = signal.butter(1, 0.1, 'low')
    yaw_data = signal.filtfilt(b, a, imu_angular_velocity[2,:])

    time_interval = encoder_stamps[1]-encoder_stamps[0]
    #消除无效range和角度
    all_ranges = lidar_ranges
    all_angles = np.arange(-135, 135.25, 0.25) * np.pi / 180.0
    start_time = tic()
    for encoder_time in range(begin_encoder_index,end_encoder_index): # 读取所有的Lidar数据
        time_lidar = np.argmin(np.abs(encoder_stamps[encoder_time] - lidar_stamps))
        angular_velocity = yaw_data[np.argmin(np.abs(imu_stamps - encoder_stamps[encoder_time]))]
        indValid = np.logical_and((all_ranges < 30), (all_ranges > 0.1))
        ranges = all_ranges[indValid[:, time_lidar]][:, time_lidar]
        angles = all_angles[indValid[:, time_lidar]]
        particles = predict(encoder_time, particles, time_interval, Number,angular_velocity)  # (X，Y) Unit:meters
        particles,weights, MAP = update(particles, weights,MAP,ranges,angles,Number)
        if neff(weights) < Number / 2:
            particles, weights = simple_resample(particles, weights)
        # 下面开始Predict和Particle Filter
        progress = 100*(encoder_time-begin_encoder_index)/(end_encoder_index-begin_encoder_index)
        print('%.2f%%' %(progress))
    toc(start_time)
    #showMap(MAP)
    plt.imshow(MAP['map'])
    np.save('map.npy',MAP['map'])
    plt.pause(0)