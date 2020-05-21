import threading
import matplotlib.pyplot as plt
import math
import time
import numpy as np
import cv2 as cv
import json

from rplidar import RPLidar, RPLidarException, MAX_MOTOR_PWM
from setting import RPLIDAR_PORT, BAUDRATE

from module.pointprocess import filter_polar_points, cvt_polar_to_cartesian
from module.pointprocess import euclidian_distant
from module.gridmap import create_occupancy_grid_map
from module.ellipse import rotate_matrix, ellipse

from tracking2 import filter_measure, normalize_weight, gaussian, check

MOTOR_PWM = MAX_MOTOR_PWM // 2
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

WINDOW_NAME = "POINTS"

ROOM_STATE = 'ROOM'
TRACK_STATE = 'TRACK'

# grid constants
gx = 200
gy = 200

# ellipse constants
a_axis = 200
b_axis = 100

# gaussian constant
g_fx = gaussian(1, 0.3)

# sampling constants
walking_speed = 700
n_sample = 200


def resampling(states, n_sample, speed):
    new_states = []
    weight = []
    max_number = 0
    for x, y, alpha, w, number in states:
        v_x, v_y = np.matmul(rotate_matrix(alpha), [0, speed])
        if v_x > 0:
            v_x_i = -v_x / 2
        else:
            v_x_i = v_x
            v_x = -v_x / 2
        if v_y > 0:
            v_y_i = -v_y / 2
        else:
            v_y_i = v_y
            v_y = -v_y / 2
        l_vector = (x + v_x_i, y + v_y_i, alpha - np.pi / 2, w, number)
        h_vector = (x + v_x, y + v_y, alpha + np.pi / 2, w, number),
        resampled = np.random.uniform(low=l_vector,
                                       high=h_vector,
                                       size=(int(n_sample * w), 5))
        for new_state in resampled:
            new_states.append(new_state)
        
        weight.append(w)
        max_number = max(number, max_number)

    m_sample = int(n_sample // 4)
    random_states = np.random.uniform(low=(-3000, -3000, 0, 1, 0),
                                   high=(3000, 3000, np.pi * 2, 0.001, 0),
                                   size=(m_sample, 5))
    random_states[:, 4] = range(int(max_number + 1), int(max_number + m_sample + 1))
    for state in random_states:
        new_states.append(state)
    new_states = np.array(new_states)
    new_states[:, 3] = new_states[:, 3] / sum(new_states[:, 3])
    return new_states


def run_RPLidar(port, baudrate):
    lidar = None

    try:
        lidar = RPLidar(port, baudrate=baudrate)
    except Exception as exception:
        print('cannot initiate RPlidar with an exception:')
        print(exception)

    if lidar:
        try:
            lidar.clear_input()

            info = lidar.get_info()
            print(info)

            health = lidar.get_health()
            print(health)

            # lidar.set_pwm(MOTOR_PWM)

            cv.namedWindow(WINDOW_NAME)

            average_time = 0
            n_time = 0

            time.sleep(5)
            loop_start_time = time.time()  # start time
            # scan (quality, angle, distance)
            #degree, mm
            print('get room points...')
            app_state = ROOM_STATE
            room_pointset = []

            ## occupancy grid map
            grid_map = None

            ## particle filter states
            states = np.random.uniform(low=(-3000, -3000, 0, 1, 0),
                                       high=(3000, 3000, np.pi * 2, 1, 0),
                                       size=(n_sample, 5))
            states[:, 4] = range(n_sample)
            state_color = dict()
            j = 0

            ## LOOP
            start_time = time.time()
            temp_points = []
            for i, scan in enumerate(
                    lidar.iter_scans(max_buf_meas=500, min_len=10)):
                new_loop_time = time.time() - start_time
                start_time = time.time()
                polar_points = filter_polar_points(scan)
                points = [
                    cvt_polar_to_cartesian(point) for point in polar_points
                ] + temp_points

                if len(points) < 150:
                    temp_points = points
                    continue
                else:
                    temp_points = []

                try:
                    ## SHARED ####
                    img = np.zeros((WIDTH, HEIGHT, 3), dtype="uint8")

                    if app_state is ROOM_STATE:
                        ## ROOM ####
                        for point in points:
                            p_x = int(np.round(point[0] * RATIO_X) + X_CENTER)
                            p_y = int(np.round(point[1] * RATIO_Y) + Y_CENTER)
                            cv.circle(img, (p_x, p_y), 1, (125, 125, 125), -1)
                            room_pointset.append(point)

                        loop_time = time.time() - loop_start_time
                        if loop_time > 30 and app_state is ROOM_STATE:
                            print('room pointset size =\t', len(room_pointset))

                            ## GRID MAP ####
                            grid_map = create_occupancy_grid_map(
                                room_pointset, 20000, 20000, (gx, gy))
                            print('grid map shape\t =', grid_map.shape)
                            cv.imshow('grid map',
                                      cv.resize(grid_map, (500, 500)))

                            app_state = TRACK_STATE

                    elif app_state is TRACK_STATE:
                        ## TRACK ###

                        speed = walking_speed * new_loop_time
                        if j > 0:
                            print('before resampling\t', states.shape)
                            states = resampling(states, n_sample, speed)
                            print('resampled ', states.shape)

                        ## calculate the weight
                        measures = filter_measure(points, grid_map, gx, gy)
                        if len(measures) > 0:
                            weight = np.array([
                                sum([
                                    w * g_fx(
                                        ellipse(a_axis, b_axis, x_0, y_0,
                                                alpha)(x, y)) * check(
                                                    (x, y), (x_0, y_0))
                                    for x, y in measures
                                ]) for x_0, y_0, alpha, w, _ in states
                            ])
                        else:
                            weight = np.ones(len(states))
                        if all(weight == 0) or sum(weight) == 0:
                            weight = np.array([1] * len(weight))
                        weight = weight / sum(weight)
                        print(states.shape)
                        states[:, 3] = weight
                        states = np.array(
                            sorted(states,
                                   key=lambda state: state[3],
                                   reverse=True))
                        states = states[:n_sample]
                        states[:, 3] = states[:, 3] / sum(states[:, 3])
                        print('weighted\t\t', min(states[:, 3]),
                              max(states[:, 3]))
                        print('states shape =\t\t', states.shape)

                        j += 1

                        for point in points:
                            p_x = int(np.round(point[0] * RATIO_X) + X_CENTER)
                            p_y = int(np.round(point[1] * RATIO_Y) + Y_CENTER)
                            cv.circle(img, (p_x, p_y), 1, (125, 125, 125), -1)

                        walking_points = np.array(measures)
                        for point in walking_points:
                            p_x = int(np.round(point[0] * RATIO_X) + X_CENTER)
                            p_y = int(np.round(point[1] * RATIO_Y) + Y_CENTER)
                            cv.circle(img, (p_x, p_y), 1, (125, 125, 0), -1)

                        n_state = 5
                        clusters = dict()

                        for x, y, alpha, _, number in states[:n_state]:
                            state_number = int(number)
                            if state_color.get(state_number) is None:
                                b, g, r = np.random.randint((50, 50, 50),
                                                            (200, 200, 200))
                                color = (int(b), int(g), int(r))
                                state_color[state_number] = color

                            if clusters.get(state_number) is None:
                                clusters[state_number] = [[x, y, alpha]]
                            else:
                                clusters[state_number].append([x, y, alpha])
                        print('estimated number of clusters =\t',
                              len(list(clusters.keys())))
                        for state_number in clusters:
                            cluster_states = np.array(clusters[state_number])
                            x_c, y_c, alpha_c = np.mean(cluster_states, axis=0)
                            p_x = int(np.round(x_c * RATIO_X) + X_CENTER)
                            p_y = int(np.round(y_c * RATIO_Y) + Y_CENTER)
                            cv.circle(img, (p_x, p_y), 1, color, -1)
                            cv.ellipse(
                                img, (p_x, p_y),
                                (int(a_axis * RATIO_X), int(b_axis * RATIO_Y)),
                                alpha_c * 180 / np.pi, 0, 360, color, 1)

                            v_x, v_y = np.matmul(rotate_matrix(alpha_c),
                                                 [0, speed]) + [x_c, y_c]
                            p_x2 = int(np.round(v_x * RATIO_X) + X_CENTER)
                            p_y2 = int(np.round(v_y * RATIO_Y) + Y_CENTER)
                            cv.arrowedLine(img, (p_x, p_y), (p_x2, p_y2),
                                           color, 1)
                        print()

                    else:
                        raise Exception('error: the state is not defined.')

                    ## SHARED ####
                    loop_time = time.time() - loop_start_time
                    cv.putText(img, 'STATE    ' + app_state, (5, HEIGHT - 5),
                               0, 0.5, (125, 125, 125))
                    cv.putText(img, 'TIME     ' + str(loop_time),
                               (5, HEIGHT - 25), 0, 0.5, (125, 125, 125))
                    cv.imshow(WINDOW_NAME, img)

                except Exception as e:
                    print('error processing points with an exception:')
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


if __name__ == '__main__':
    run_RPLidar(RPLIDAR_PORT, BAUDRATE)
