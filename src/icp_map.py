import signal
import sys
import time
import threading
import numpy as np
import cv2

from rplidar import RPLidar, RPLidarException

from module.pointprocess import filter_polar_points, cvt_polar_to_cartesian
from module.icp import tr_icp, icp, closest_points
from module.gridmap import create_grid_map, grid_mapping

# MOTOR_PWM = MAX_MOTOR_PWM // 2
# MAX_BUFF = 500
# MIN_LEN = 10

WIDTH = 10000
HEIGHT = 10000
X_CENTER = WIDTH // 2
Y_CENTER = HEIGHT // 2

X_MAX = 5000
Y_MAX = 5000

RATIO_X = WIDTH / (X_MAX * 2)
RATIO_Y = HEIGHT / (Y_MAX * 2)

gx = 50
gy = 50

WINDOW_NAME = "POINTS"
WINDOW_NAME2 = "GRID"

scan_output = []
new_scan = False
event = threading.Event()


def thread_func():

    cv2.namedWindow(WINDOW_NAME)
    prev_points = np.array([])
    grid_map = create_grid_map(250, 250)
    
    R = np.eye(2)
    T = np.array([0, 0])
    N = 100

    global scan_output
    global new_scan

    while True:
        if new_scan:
            img = np.zeros((WIDTH, HEIGHT, 3), dtype="uint8")
            polar_points = filter_polar_points(scan_output)
            new_scan = False
            points = np.array(
                [cvt_polar_to_cartesian(point) for point in polar_points]
            )
            points = np.matmul(R, points.T).T + T

            
            for point in points[- N:]:
                p_x = int(np.round(point[0] * RATIO_X) + X_CENTER)
                p_y = int(np.round(point[1] * RATIO_Y) + Y_CENTER)
                cv2.circle(img, (p_x, p_y), 15, (0, 0, 205), -1)
            for point in prev_points:
                p_x = int(np.round(point[0] * RATIO_X) + X_CENTER)
                p_y = int(np.round(point[1] * RATIO_Y) + Y_CENTER)
                cv2.circle(img, (p_x, p_y), 15, (125, 125, 125), -1)
            cv2.imshow(WINDOW_NAME, cv2.resize(img, (500, 500)))

            n_p = prev_points.shape[0]
            if n_p > 100:
                X = closest_points(points[- N:], prev_points)
                err = np.sum(
                    np.sqrt(np.sum(
                        (X - points[- N:])**2, axis=1))) /N
                if err > 0:
                    print('prev err =\t\t', err)
                    r, t = icp(prev_points, points[- N:], N_iter=10)
                    # r, t = tr_icp(prev_points[: N], points[: N], N=20, N_iter=100)
                    points = np.matmul(r, points.T).T + t
                    X = closest_points(points[- N:], prev_points)
                    err = np.sum(
                        np.sqrt(np.sum(
                            (X - points[- N:])**2, axis=1))) / N
                    print('new err =\t\t', err)

                    R = np.matmul(r, R)
                    T = np.matmul(r, T) + t

            grid_map = grid_mapping(grid_map, points, gx, gy)

            for point in points[- N:]:
                p_x = int(np.round(point[0] * RATIO_X) + X_CENTER)
                p_y = int(np.round(point[1] * RATIO_Y) + Y_CENTER)
                cv2.circle(img, (p_x, p_y), 15, (125, 125, 50), -1)
            prev_points = points.copy()[-200:]
            cv2.imshow(WINDOW_NAME, cv2.resize(img, (500, 500)))
            cv2.imshow(WINDOW_NAME2, cv2.resize(grid_map, (500, 500)))

        if cv2.waitKey(2) == 27:
            event.set()
            break
        
        if  event.is_set():
            break


def main():

    lidar = None
    try:
        lidar = RPLidar('COM3')
    except Exception as exception:
        print('cannot initiate RPlidar with an exception:')
        print(exception)

    def keyboardInterruptHandler(signal, frame):
        print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".
              format(signal))
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
        event.set()
        exit(0)

    signal.signal(signal.SIGINT, keyboardInterruptHandler)

    try:
        if lidar is not None:

            try:
                info = lidar.get_info()
                print(info)
            except:
                print(sys.exc_info())

            health = lidar.get_health()
            print(health)

            iterator = lidar.iter_scans()

            time.sleep(5)
            # LOOP BEGIN
            t = threading.Thread(target=thread_func)
            t.start()
            global scan_output
            global new_scan
            scan_read = []
            while True:
                scan_read += next(iterator)
                if len(scan_read) > 200:
                    scan_output = scan_read
                    new_scan = True
                    # print('scan output', len(scan_output))
                    scan_read = []
                if event.is_set():
                    break

            # LOOP END

            lidar.stop()
            lidar.stop_motor()
            lidar.disconnect()
    except RPLidarException:
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
        event.set()
        print(sys.exc_info())


if __name__ == "__main__":
    main()
