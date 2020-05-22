import numpy as np
import cv2
from sklearn.cluster import MeanShift
from module.data import load_to_dicts
from module.rplidarx import cvtPolarToCartesian
from module.gridmap import create_occupancy_grid_map
from module.ellipse import rotate_matrix, ellipse
from module.disjoint_set import DisjointSet
from module.pointprocess import euclidian_distant


def get_frames(point_stream, field_names):
    point_frames = []
    point_frame = []
    for point_dict in point_stream:
        if 'newscan' in field_names and point_dict['newscan'] > 0:
            if len(point_frame) > 0:
                point_frames.append(point_frame)
                point_frame = []
        if 'angle' in field_names and 'distance' in field_names:
            angle = point_dict.get('angle')
            distant = point_dict.get('distance')

            angle = angle % 360
            if distant >= 10:
                cpoint = cvtPolarToCartesian([angle, distant])
                point_frame.append(cpoint)

    return point_frames


def filter_measure(points, grid_map, gx, gy):
    h, w, _ = grid_map.shape
    x0, y0 = w // 2, h // 2
    measure = []
    for x, y in points:
        xi = int(np.floor(x / gx) + x0)
        yi = int(np.floor(y / gy) + y0)
        if xi < w and yi < h:
            is_background = grid_map[h - yi - 1, xi, 1]
            is_not_background = grid_map[h - yi - 1, xi, 2]
            total = is_background + is_not_background
            prob = is_not_background / total if total > 0 else 0.5
            if prob > 0.9:
                measure.append([x, y])
    return np.array(measure)


def normalize_weight(weighted_points):
    total_weight = sum(w for _, w in weighted_points)
    return [[new_state, w / total_weight] for new_state, w in weighted_points]


def gaussian(mu, sig):
    def func(x):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    return func


def p(Z, sig):
    def given(x, func):
        return sum(gaussian(mu=0, sig=sig)(func(x, z)) for z in Z)

    return given


def check(point_a, point_b):
    a = np.array(point_a)
    b = np.array(point_b)
    return np.linalg.norm(a) <= np.linalg.norm(b)


if __name__ == '__main__':

    room_path = 'data/nut/room.csv'
    walking_path = 'data/nut/straight.csv'

    ## BACKGROUND ################################################################
    ROOM_FIELD_NAMES, ROOM_POINT_STREAM = load_to_dicts(room_path)
    ROOM_FRAMES = get_frames(ROOM_POINT_STREAM, ROOM_FIELD_NAMES)
    print('length of stream\t\t', len(ROOM_POINT_STREAM))
    print('total number of frames\t', len(ROOM_FRAMES))

    # for frame in ROOM_FRAMES:
    #     points = np.array(frame)
    #     plt.scatter(points[:, 0], points[:, 1], s=2, c='k', alpha=0.1)
    # plt.show()

    ROOM_POINTS = []
    for points in ROOM_FRAMES:
        for point in points:
            ROOM_POINTS.append(point)
    ROOM_POINTS = np.array(ROOM_POINTS)

    gx = 50
    gy = 50
    # GMAP = create_grid_map(gx, gy, (10000, 10000))
    N_FRAME = 100
    POINTS_FOR_GRID_MAP = []
    for points in ROOM_FRAMES[0:N_FRAME]:
        for point in points:
            POINTS_FOR_GRID_MAP.append(point)
    GMAP = create_occupancy_grid_map(POINTS_FOR_GRID_MAP, 10000, 10000,
                                     (gx, gy))
    print('shape\t\t\t', GMAP.shape)
    print('number of frames\t', N_FRAME if N_FRAME > 0 else len(ROOM_FRAMES))
    cv2.imshow('grid map', cv2.resize(GMAP, (500, 500)))

    ## WALKING #################################################################
    WALKING_FIELD_NAMES, WALKING_POINT_STREAM = load_to_dicts(walking_path)
    WALKING_FRAMES = get_frames(WALKING_POINT_STREAM, WALKING_FIELD_NAMES)[1:]
    print('length of stream\t', len(WALKING_POINT_STREAM))
    print('total number of frames\t', len(WALKING_FRAMES))

    # ellipse constant
    a_axis = 200
    b_axis = 100

    # gaussian constant
    g_fx = gaussian(1, 0.3)

    total_time = 30  # s
    N_WALKING_FRAME = 100
    speed = 1400 * total_time / N_WALKING_FRAME

    n_sample = 200

    states = np.random.uniform(low=(-3000, -3000, 0, 1, 0),
                               high=(3000, 3000, np.pi * 2, 1, 0),
                               size=(n_sample, 5))
    states[:, 4] = range(n_sample)
    state_color = dict()

    WIDTH = 1000
    HEIGHT = 1000
    X_CENTER = WIDTH // 2
    Y_CENTER = HEIGHT // 2

    X_MAX = 5000
    Y_MAX = 5000

    RATIO_X = WIDTH / (X_MAX * 2)
    RATIO_Y = HEIGHT / (Y_MAX * 2)
    print('ratio =\t\t', RATIO_X, RATIO_Y)

    WINDOW_NAME = "POINTS"

    i = 0
    for points in WALKING_FRAMES[:N_WALKING_FRAME]:
        if i > 0:
            print('before resampling\t', states.shape)
            new_states = []
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
                resampling = np.random.uniform(low=l_vector,
                                               high=h_vector,
                                               size=(int(n_sample * w), 5))
                for new_state in resampling:
                    new_states.append(new_state)
            states = np.array(new_states)
            print('resampled ', states.shape)

        measures = filter_measure(points, GMAP, gx, gy)
        if len(measures) > 0:
            weight = np.array([
                sum([
                    w * g_fx(ellipse(a_axis, b_axis, x_0, y_0, alpha)(x, y)) *
                    check((x, y), (x_0, y_0)) for x, y in measures
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
            sorted(states, key=lambda state: state[3], reverse=True))
        states = states[:n_sample]
        states[:, 3] = states[:, 3] / sum(states[:, 3])
        print('weighted\t\t', min(states[:, 3]), max(states[:, 3]))
        print('states shape =\t\t', states.shape)

        # plot
        img = np.zeros((WIDTH, HEIGHT, 3), dtype="uint8")
        for point in points:
            p_x = int(np.round(point[0] * RATIO_X) + X_CENTER)
            p_y = int(np.round(point[1] * RATIO_Y) + Y_CENTER)
            cv2.circle(img, (p_x, p_y), 1, (125, 125, 125), -1)

        walking_points = np.array(measures)
        for point in walking_points:
            p_x = int(np.round(point[0] * RATIO_X) + X_CENTER)
            p_y = int(np.round(point[1] * RATIO_Y) + Y_CENTER)
            cv2.circle(img, (p_x, p_y), 1, (125, 125, 0), -1)

        clusters = DisjointSet(list(range(len(states))))
        for i, s1 in enumerate(states[:, : 3]):
            for j, s2 in enumerate(states[i + 1:, : 3]):
                x1, y1, _ = s1
                x2, y2, _ = s2
                dist = euclidian_distant([x1, y1], [x2, y2])
                if dist <= 1000:
                    clusters.union(i, j)

        # print('cluster =\t', len(clusters.get()))
        color = (125, 50, 125)
        min_cluster_state = 30
        for cluster_states in clusters.get():
            if len(cluster_states) >= min_cluster_state:
                cluster = np.array([states[i, : 4] for i in cluster_states])
                x_c, y_c, alpha_c = np.average(cluster[:, : 3], axis=0, weights=cluster[:, 3])
                p_x = int(np.round(x_c * RATIO_X) + X_CENTER)
                p_y = int(np.round(y_c * RATIO_Y) + Y_CENTER)
                cv2.circle(img, (p_x, p_y), 1, color, -1)
                cv2.ellipse(img, (p_x, p_y),
                            (int(a_axis * RATIO_X), int(b_axis * RATIO_Y)),
                            alpha_c * 180 / np.pi, 0, 360, color, 1)

                v_x, v_y = np.matmul(rotate_matrix(alpha_c),
                                    [0, speed]) + [x_c, y_c]
                p_x2 = int(np.round(v_x * RATIO_X) + X_CENTER)
                p_y2 = int(np.round(v_y * RATIO_Y) + Y_CENTER)
                cv2.arrowedLine(img, (p_x, p_y), (p_x2, p_y2), color, 1)
        
        # n_state = -1
        # clusters = dict()

        # labels = MeanShift().fit_predict(states[:, :2])

        # state_dict = dict()
        # for i_s, s in enumerate(states):
        #     c_i = labels[i_s]
        #     s_i = s[: 3]
        #     if state_dict.get(c_i) is None:
        #         state_dict[c_i] = [s_i]
        #     else:
        #         state_dict[c_i].append(s_i)

        # pred_states = [np.mean(state_dict[i_d], axis=0) for i_d in state_dict]

        # color = (125, 50, 125)
        # for x_c, y_c, alpha_c in pred_states:
        #     p_x = int(np.round(x_c * RATIO_X) + X_CENTER)
        #     p_y = int(np.round(y_c * RATIO_Y) + Y_CENTER)
        #     cv2.circle(img, (p_x, p_y), 1, color, -1)
        #     cv2.ellipse(
        #         img, (p_x, p_y),
        #         (int(a_axis * RATIO_X), int(b_axis * RATIO_Y)),
        #         alpha_c * 180 / np.pi, 0, 360, color, 1)

        #     v_x, v_y = np.matmul(rotate_matrix(alpha_c),
        #                             [0, speed]) + [x_c, y_c]
        #     p_x2 = int(np.round(v_x * RATIO_X) + X_CENTER)
        #     p_y2 = int(np.round(v_y * RATIO_Y) + Y_CENTER)
        #     cv2.arrowedLine(img, (p_x, p_y), (p_x2, p_y2),
        #                     color, 1)

        cv2.imshow(WINDOW_NAME, img)
        if cv2.waitKey(int(total_time / N_WALKING_FRAME * 1000)) > -1:
            break

        print()
        i += 1

    print('end')
    cv2.waitKey(0)
