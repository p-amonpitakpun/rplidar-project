import sys
import time
import numpy as np
import cv2
import json

from rplidar import RPLidar, RPLidarException, MAX_MOTOR_PWM
from setting import RPLIDAR_PORT, BAUDRATE

from module.pointprocess import filter_polar_points, cvt_polar_to_cartesian
from module.pointprocess import euclidian_distant
from module.gridmap import create_occupancy_grid_map
from module.ellipse import rotate_matrix, ellipse
from module.disjoint_set import DisjointSet

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
gx = 50
gy = 50

# ellipse constants
a_axis = 200
b_axis = 100

# gaussian constant
g_fx = gaussian(1, 0.6)

# sampling constants
walking_speed = 700
n_sample = 200

radius = 3000


def resampling(states, n_sample, speed):
    new_states = []
    weight = []
    use_states = states[:n_sample]
    use_states[:, 3] = use_states[:, 3] / sum(use_states[:, 3])
    for x, y, alpha, w in use_states:
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
        l_vector = (x + v_x_i, y + v_y_i, alpha - np.pi / 2, 1)
        h_vector = (x + v_x, y + v_y, alpha + np.pi / 2, 1),
        resampled = np.random.uniform(low=l_vector,
                                      high=h_vector,
                                      size=(int(n_sample * w), 4))
        for new_state in resampled:
            new_states.append(new_state)

        weight.append(w)

    m_sample = int(n_sample)
    random_states = np.random.uniform(low=(-radius, -radius, 0, 1),
                                      high=(radius, radius, np.pi * 2, 1),
                                      size=(m_sample, 4))
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
        # try:
        if 1:
            lidar.clear_input()

            info = lidar.get_info()
            print(info)

            health = lidar.get_health()
            print(health)

            # lidar.set_pwm(MOTOR_PWM)

            cv2.namedWindow(WINDOW_NAME)

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
            states = np.random.uniform(low=(-radius, -radius, 0, 1),
                                       high=(radius, radius, np.pi * 2, 1),
                                       size=(n_sample, 4))
            state_color = dict()
            first_states = True

            ## LOOP
            start_time = time.time()
            temp_points = []
            for i_s, scan in enumerate(
                    lidar.iter_scans(max_buf_meas=500, min_len=10)):
                new_loop_time = time.time() - start_time
                start_time = time.time()
                polar_points = filter_polar_points(scan)
                points = [
                    cvt_polar_to_cartesian(point) for point in polar_points
                ] + temp_points

                if len(points) < 150:
                    temp_points = points
                else:
                    temp_points = []
                    # try:
                    if 1:
                        ## SHARED ####
                        img = np.zeros((WIDTH, HEIGHT, 3), dtype="uint8")

                        if app_state is ROOM_STATE:
                            ## ROOM ####
                            for point in points:
                                p_x = int(
                                    np.round(point[0] * RATIO_X) + X_CENTER)
                                p_y = int(
                                    np.round(point[1] * RATIO_Y) + Y_CENTER)
                                cv2.circle(img, (p_x, p_y), 1, (125, 125, 125),
                                           -1)
                                room_pointset.append(point)

                            loop_time = time.time() - loop_start_time
                            if loop_time > 10 and app_state is ROOM_STATE:
                                print('room pointset size =\t',
                                      len(room_pointset))

                                ## GRID MAP ####
                                grid_map = create_occupancy_grid_map(
                                    room_pointset, 20000, 20000, (gx, gy))
                                print('grid map shape\t =', grid_map.shape)
                                cv2.imshow('grid map',
                                           cv2.resize(grid_map, (500, 500)))

                                app_state = TRACK_STATE

                        elif app_state is TRACK_STATE:
                            ## TRACK ###

                            speed = walking_speed  #* new_loop_time
                            if not first_states:
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
                                    ]) for x_0, y_0, alpha, w in states
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
                            print('states shape =\t\t', states.shape)

                            first_states = False

                            for point in points:
                                p_x = int(
                                    np.round(point[0] * RATIO_X) + X_CENTER)
                                p_y = int(
                                    np.round(point[1] * RATIO_Y) + Y_CENTER)
                                if point in measures:
                                    cv2.circle(img, (p_x, p_y), 1,
                                               (125, 125, 0), -1)
                                else:
                                    cv2.circle(img, (p_x, p_y), 1,
                                               (125, 125, 125), -1)

                            # for point in states:
                            #     p_x = int(np.round(point[0] * RATIO_X) + X_CENTER)
                            #     p_y = int(np.round(point[1] * RATIO_Y) + Y_CENTER)
                            #     cv2.circle(img, (p_x, p_y), 1, (20, 75, 75), -1)

                            ## clustering ##################################
                            clusters = DisjointSet(list(range(len(states))))
                            for i, s1 in enumerate(states[:, :3]):
                                for j, s2 in enumerate(states[i + 1:, :3]):
                                    x1, y1, _ = s1
                                    x2, y2, _ = s2
                                    dist = euclidian_distant([x1, y1],
                                                             [x2, y2])
                                    if dist <= 500:
                                        clusters.union(i, j)

                            # print('cluster =\t', len(clusters.get()))
                            color = (125, 50, 125)
                            min_cluster_state = 1
                            max_cluster_len = 200
                            for cluster_states in clusters.get():
                                max_cluster_len = max(max_cluster_len,
                                                      len(cluster_states))
                                if len(cluster_states) >= min_cluster_state:
                                    cluster = np.array(
                                        [states[i] for i in cluster_states])
                                    w = cluster[:, 3]
                                    if sum(w) > 0:
                                        x_c, y_c, alpha_c = np.average(
                                            cluster[:, :3], axis=0, weights=w)
                                        p_x = int(
                                            np.round(x_c * RATIO_X) + X_CENTER)
                                        p_y = int(
                                            np.round(y_c * RATIO_Y) + Y_CENTER)
                                        cv2.circle(img, (p_x, p_y), 1, color,
                                                   -1)
                                        cv2.ellipse(img, (p_x, p_y),
                                                    (int(a_axis * RATIO_X),
                                                     int(b_axis * RATIO_Y)),
                                                    alpha_c * 180 / np.pi, 0,
                                                    360, color, 1)

                                        v_x, v_y = np.matmul(
                                            rotate_matrix(alpha_c),
                                            [0, speed]) + [x_c, y_c]
                                        p_x2 = int(
                                            np.round(v_x * RATIO_X) + X_CENTER)
                                        p_y2 = int(
                                            np.round(v_y * RATIO_Y) + Y_CENTER)
                                        cv2.arrowedLine(
                                            img, (p_x, p_y), (p_x2, p_y2),
                                            color, 1)
                            print('n cluster =\t\t', len(clusters.get()))
                            print(
                                'zero cluster =\t\t',
                                sum([
                                    1 if sum(
                                        [states[i][3]
                                         for i in state_indices]) == 0 else 0
                                    for state_indices in clusters.get()
                                ]))
                            print('max cluster =\t\t', max_cluster_len)
                            print()

                        else:
                            raise Exception('error: the state is not defined.')

                        ## SHARED ####
                        loop_time = time.time() - loop_start_time
                        cv2.putText(img, 'STATE    ' + app_state,
                                    (5, HEIGHT - 5), 0, 0.5, (125, 125, 125))
                        cv2.putText(img, 'TIME     ' + str(loop_time),
                                    (5, HEIGHT - 25), 0, 0.5, (125, 125, 125))
                        cv2.imshow(WINDOW_NAME, img)

                    # except Exception as e:
                    #     print('error processing points with an exception:')
                    #     print('\t', e)
                    #     print('\t', sys.exc_info())
                    #     break

                    if cv2.waitKey(5) > -1:
                        break

        # except Exception as e:
        #     print("error")
        #     print(e)

        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_RPLidar(RPLIDAR_PORT, BAUDRATE)
