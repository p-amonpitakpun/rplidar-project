from rplidar import RPLidar, RPLidarException, MAX_MOTOR_PWM
import threading
import matplotlib.pyplot as plt
import math
import time
import numpy as np
import cv2 as cv

from setting import RPLIDAR_PORT


def draw():
    global is_plot
    while is_plot:
        w = 10000
        h = 10000
        mat = np.zeros((w, h, 1), dtype="uint8")
        plt.figure(1)
        plt.cla()
        plt.ylim(0, 10000)
        plt.xlim(0, 10000)
        if (len(x) >= 0 and len(y) >= 0):
            X = np.array(x) + 5000
            Y = np.array(y) + 5000
            plt.scatter(X, Y, c='k', s=1)
        plt.pause(0.001)


is_plot = True
x = []
y = []

lidar = None
try:
    lidar = RPLidar(RPLIDAR_PORT)
except Exception as e:
    print(e)
    exit(1)
if RPLidar:
    try:
        threading.Thread(target=draw).start()

        lidar.clear_input()

        info = lidar.get_info()
        print(info)

        health = lidar.get_health()
        print(health)

        lidar.set_pwm(MAX_MOTOR_PWM * 3 // 4)

        time.sleep(5)
        t = time.time()  # start time
        # scan (quality, angle, distance)
        #degree, mm
        for i, scan in enumerate(lidar.iter_scans(max_buf_meas=1000)):
            print('%d: Got %d measurments' % (i, len(scan)))

            x = []
            y = []
            for m in scan:
                s_distance = m[2]
                s_angle = math.floor(m[1])

                if s_angle <= 0:
                    s_angle = 0
                elif s_angle >= 359:
                    s_angle = 359

                if (s_distance > 10):
                    x.append(s_distance * math.cos(math.radians(s_angle)))
                    y.append(s_distance * math.sin(math.radians(s_angle)))

        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
    except Exception as e:
        print("error")
        print(e)
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()

    is_plot = False
