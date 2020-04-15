import csv
import threading
import matplotlib.pyplot as plt
import math
import time
import numpy as np
import cv2 as cv
import json
import signal
import os

from rplidar import RPLidar, RPLidarException, MAX_MOTOR_PWM
from setting import RPLIDAR_PORT, BAUDRATE

MOTOR_PWM = MAX_MOTOR_PWM // 4
MAX_BUFF = 500
MIN_LEN = 10

WIDTH = 1000
HEIGHT = 1000
X_CENTER = WIDTH // 2
Y_CENTER = HEIGHT // 2

X_MAX = 5000
Y_MAX = 5000

RATIO_X = WIDTH / (X_MAX * 2)
RATIO_Y = HEIGHT / (Y_MAX * 2)


def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(
        signal))
    exit(0)


def limitPolarPoints(scanPoints):
    LOWER_DEGREE_LIMIT = 0
    UPPER_DEGREE_LIMIT = 359.99
    LOWER_DISTANT_LIMIT = 10

    new_points = []
    for point in scanPoints:
        degree, distance = point
        if degree >= LOWER_DEGREE_LIMIT and degree <= UPPER_DEGREE_LIMIT and distance >= LOWER_DISTANT_LIMIT:
            new_points.append((degree, distance))
    return new_points


def cvtPolarToCartesian(polarPoint):
    degree, distance = polarPoint
    rad = np.radians(degree)
    return (distance * np.cos(rad), distance * np.sin(rad))


def mapPointToMat(point):
    return (int(np.round(point[0] * RATIO_X) + X_CENTER),
            int(np.round(point[1] * RATIO_Y) + Y_CENTER))


def run_RPLidar(port, baudrate, dat, config):

    signal.signal(signal.SIGINT, keyboardInterruptHandler)

    lidar = None
    csvfile = None
    writer = None
    try:
        lidar = RPLidar(port, baudrate=baudrate)
    except Exception as e:
        print(e)
    if lidar:
        try:
            lidar.clear_input()

            info = lidar.get_info()
            print(info)

            health = lidar.get_health()
            print(health)

            lidar.set_pwm(MOTOR_PWM)

            time.sleep(5)
            ##########################################################################
            path = 'data/stream/example.csv'
            endtime = 30
            points = []
            print('write stream to', path)
            with open(path, 'w', newline='\r\n') as csvfile:
                fieldnames = ['newscan', 'angle', 'distant', 'quality']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                print('start writing')
                begin = time.time()
                for i, scan in enumerate(
                        lidar.iter_scans(max_buf_meas=500, min_len=10)):
                    try:
                        now = time.time()
                        if now - begin > endtime: break
                        newscan = 1
                        for point in scan:
                            quality, angle, distant, = point
                            writer.writerow({
                                'newscan': newscan,
                                'angle': angle,
                                'distant': distant,
                                'quality': quality
                            })
                            newscan = 0

                            polar = limitPolarPoints([[angle, distant]])
                            cartesian = cvtPolarToCartesian(polar[0])
                            points.append(cartesian)
                    except Exception as e:
                        print(e)
                        break
                print('finished writing')
                print('written', os.stat(path).st_size, 'bytes')
            
            srcMat = np.zeros((WIDTH, HEIGHT, 3), dtype="uint8")
            matPoints = list(map(mapPointToMat, points))
            for center in matPoints:
                cv.circle(srcMat, center, 1, (255, 255, 255), -1)
            cv.imshow(path, srcMat)
            print('press any key to continue')
            cv.waitKey(0)
            ##########################################################################
        except Exception as e:
            print("error")
            print(e)

        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()


if __name__ == '__main__':
    run_RPLidar(RPLIDAR_PORT, BAUDRATE, None, None)
