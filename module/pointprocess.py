import matplotlib.pyplot as plt
import math
import time
import numpy as np
import cv2 as cv

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

def findDistant(pointA, pointB):
    pointA = np.array(pointA)
    pointB = np.array(pointB)
    return np.sqrt(np.sum(np.power(pointA - pointB, 2)))

def detectConvex(cPoints, prvCnt):
    srcMat = np.zeros((WIDTH, HEIGHT, 3), dtype="uint8")
    matPoints = list(map(mapPointToMat, cPoints))
    dstMat = srcMat.copy()

    if not (prvCnt is None):
        cv.drawContours(dstMat, [prvCnt], -1, (0, 125, 125), 1)
    cnt = np.array(matPoints)
    for center in matPoints:
        cv.circle(dstMat, center, 1, (255, 255, 255), -1)
    cv.drawContours(dstMat, [cnt], -1, (125, 125, 125), 1)
    approx = cv.approxPolyDP(cnt, 3, True)
    # for center in approx:
    #     cv.circle(dstMat, tuple(center[0]), 1, (255, 255, 255), -1)
    # cv.drawContours(dstMat, [approx], -1, (0, 0, 255), 1)
    hull = cv.convexHull(approx)
    cv.drawContours(dstMat, [hull], -1, (255, 0, 0), 1)
    hull = cv.convexHull(approx, returnPoints=False)
    defects = cv.convexityDefects(approx, hull)
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        far = tuple(approx[f][0])
        area = cv.contourArea(approx[s:e + 1])
        if area > 4000:
            cv.circle(dstMat, far, 5, (0, 255, 255), -1)
            cv.putText(dstMat, str(far), far, 0, 0.5, (255, 255, 0))
    return dstMat, cnt


def run_pointprocess(dat, config):

    cv.namedWindow(pointWindowName)
    prvCnt = None

    while True:
        cPoints = dat.get('p')
        if not (cPoints is None):
            try:
                dstMat, prvCnt = detectConvex(cPoints, prvCnt)
                cv.imshow(pointWindowName, dstMat)
            except Exception as e:
                print(e)
                break
            if cv.waitKey(10) > -1:
                break
    cv.destroyAllWindows()