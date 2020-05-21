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

WINDOW_NAME = "POINTS"


def filter_polar_points(scan_points):
    lower_degree_limit = 0
    upper_degree_limit = 359.99
    lower_distant_limit = 10

    new_points = []
    for point in scan_points:
        _, degree, distance = point
        if ((degree >= lower_degree_limit) and (degree <= upper_degree_limit)
                and (distance >= lower_distant_limit)):
            new_points.append((degree, distance))
    return new_points


def cvt_polar_to_cartesian(polar_point):
    degree, distance = polar_point
    rad = np.radians(degree)
    return (distance * np.cos(rad), distance * np.sin(rad))


def map_point_to_mat(point):
    return (int(np.round(point[0] * RATIO_X) + X_CENTER),
            int(np.round(point[1] * RATIO_Y) + Y_CENTER))


def euclidian_distant(point1, point2):

    assert len(point1) == len(point2)
    return np.sqrt(sum([(x1 - x2)**2 for x1, x2 in zip(point1, point2)]))


def detect_convex(cpoints, prvCnt):
    src = np.zeros((WIDTH, HEIGHT, 3), dtype="uint8")
    matPoints = list(map(map_point_to_mat, cpoints))
    dst = src.copy()

    if not (prvCnt is None):
        cv.drawContours(dst, [prvCnt], -1, (0, 125, 125), 1)
    cnt = np.array(matPoints)
    for center in matPoints:
        cv.circle(dst, center, 1, (255, 255, 255), -1)
    cv.drawContours(dst, [cnt], -1, (125, 125, 125), 1)
    approx = cv.approxPolyDP(cnt, 3, True)
    # for center in approx:
    #     cv.circle(dstMat, tuple(center[0]), 1, (255, 255, 255), -1)
    # cv.drawContours(dstMat, [approx], -1, (0, 0, 255), 1)
    hull = cv.convexHull(approx)
    cv.drawContours(dst, [hull], -1, (255, 0, 0), 1)
    hull = cv.convexHull(approx, returnPoints=False)
    defects = cv.convexityDefects(approx, hull)
    for i in range(defects.shape[0]):
        s, e, f, _ = defects[i, 0]
        far = tuple(approx[f][0])
        area = cv.contourArea(approx[s:e + 1])
        if area > 4000:
            cv.circle(dst, far, 5, (0, 255, 255), -1)
            cv.putText(dst,
                       str(e - s) + ',' + str(far), far, 0, 0.5, (255, 255, 0))
    return dst, cnt


def run_pointprocess(dat, config):

    cv.namedWindow(WINDOW_NAME)
    prvCnt = None

    while True:
        cPoints = dat.get('p')
        if not (cPoints is None):
            try:
                dstMat, prvCnt = detect_convex(cPoints, prvCnt)
                cv.imshow(WINDOW_NAME, dstMat)
            except Exception as e:
                print(e)
                break
            if cv.waitKey(20) > -1 or config.get('err') == 1:
                break
    cv.destroyAllWindows()
    print('end presentation')
