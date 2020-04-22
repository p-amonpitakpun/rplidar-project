import threading
import matplotlib.pyplot as plt
import math
import time
import numpy as np
import cv2 as cv
import json

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

pointWindowName = "POINTS"


def limitPolarPoints(scanPoints):
    LOWER_DEGREE_LIMIT = 0
    UPPER_DEGREE_LIMIT = 359.99
    LOWER_DISTANT_LIMIT = 10

    new_points = []
    for point in scanPoints:
        _, degree, distance = point
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


def process(cPoints):
    src = np.zeros((WIDTH, HEIGHT, 3), dtype="uint8")
    mat_points = list(map(mapPointToMat, cPoints))
    for center in mat_points:
        cv.circle(src, center, 2, (255, 255, 255), -1)
    dst = src.copy()

    cnt = np.array(mat_points)
    cv.drawContours(dst, [cnt], -1, (255, 255, 255), 1)

    hull = cv.convexHull(cnt)
    cv.drawContours(dst, [hull], -1, (255, 0, 0), 1)
    hull = cv.convexHull(cnt, returnPoints=False)
    defects = cv.convexityDefects(cnt, hull)
    for i in range(defects.shape[0]):
        _, _, f, d = defects[i, 0]
        far = tuple(cnt[f])
        if d > 10000:
            cv.circle(dst, far, 5, (0, 255, 125), -1)
    cv.imshow(pointWindowName, dst)


def run_RPLidar(port, baudrate, dat, config):
    lidar = None
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

            cv.namedWindow(pointWindowName)

            average_time = 0
            n_time = 0

            time.sleep(5)
            t = time.time()  # start time
            # scan (quality, angle, distance)
            #degree, mm
            for i, scan in enumerate(
                    lidar.iter_scans(max_buf_meas=500, min_len=10)):
                start_time = time.time()
                polarPoints = limitPolarPoints(scan)
                cPoints = list(map(cvtPolarToCartesian, polarPoints))

                try:
                    process(cPoints)
                    with open('./out/scan.json', 'w') as scanfile:
                        json.dump(cPoints, scanfile)
                except Exception as e:
                    print(e)
                    break
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 10**3
                average_time = (average_time * n_time +
                                elapsed_time) / (n_time + 1)
                n_time += 1
                et_str = str(elapsed_time)[:str(elapsed_time).find('.') + 4]
                # print(i, ': Got', len(scan), 'measurments', et_str, 'ms elasped')
                if cv.waitKey(10) > -1:
                    break
        except Exception as e:
            print("error")
            print(e)

        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()

    cv.destroyAllWindows()

    print("average elapsed time =", average_time, "ms")


if __name__ == '__main__':
    run_RPLidar(RPLIDAR_PORT, BAUDRATE, None, None)
