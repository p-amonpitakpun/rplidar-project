import threading
import matplotlib.pyplot as plt
import math
import time
import numpy as np
import cv2 as cv

from rplidar import RPLidar, RPLidarException, MAX_MOTOR_PWM
from setting import RPLIDAR_PORT

WIDTH = 1000
HEIGHT = 1000
X_CENTER = WIDTH // 2
Y_CENTER = HEIGHT // 2

X_MAX = 5000
Y_MAX = 5000

RATIO_X = WIDTH / (X_MAX * 2)
RATIO_Y = HEIGHT / (Y_MAX * 2)


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


if __name__ == '__main__':
    lidar = None
    try:
        lidar = RPLidar(RPLIDAR_PORT)
    except Exception as e:
        print(e)
    if lidar:
        try:
            lidar.clear_input()

            info = lidar.get_info()
            print(info)

            health = lidar.get_health()
            print(health)

            lidar.set_pwm(MAX_MOTOR_PWM)

            pointWindowName = "POINTS"
            cv.namedWindow(pointWindowName)

            time.sleep(5)
            t = time.time()  # start time
            # scan (quality, angle, distance)
            #degree, mm
            for i, scan in enumerate(lidar.iter_scans(max_buf_meas=1000)):
                # print('%d: Got %d measurments' % (i, len(scan)))
                polarPoints = limitPolarPoints(scan)
                cPoints = list(map(cvtPolarToCartesian, polarPoints))

                try:
                    srcMat = np.zeros((WIDTH, HEIGHT, 3), dtype="uint8")
                    matPoints = list(map(mapPointToMat, cPoints))
                    for center in matPoints:
                        cv.circle(srcMat, center, 2, (255, 255, 255), -1)

                    kernel = np.ones((5, 5), np.uint8)
                    dstMat = srcMat.copy()
                    # dstMat = cv.erode(dstMat, kernel, iterations=1)

                    gray = cv.cvtColor(dstMat, cv.COLOR_BGR2GRAY)
                    contours, hierarchy = cv.findContours(
                        gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                    cnt = np.array(matPoints)
                    cv.drawContours(dstMat, [cnt], -1, (0, 0, 255), 1)

                    hull = cv.convexHull(cnt, returnPoints=False)
                    defects = cv.convexityDefects(cnt, hull)
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        far = tuple(cnt[f])
                        if d > 10000:
                            cv.circle(dstMat, far, 5, (0, 255, 125), -1)

                    cv.imshow(pointWindowName, dstMat)

                except Exception as e:
                    print(e)
                    break
                if cv.waitKey(10) > -1:
                    break
        except Exception as e:
            print("error")
            print(e)

        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()

    cv.destroyAllWindows()