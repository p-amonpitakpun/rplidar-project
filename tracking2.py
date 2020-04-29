import matplotlib.pyplot as plt
import numpy as np
from module.data import load_to_dicts
from module.rplidarx import cvtPolarToCartesian
from module.pointprocess import find_distant
from module.gridmap import create_occupancy_grid_map


def get_frames(point_stream, field_names):
    point_frames = []
    point_frame = []
    for point_dict in point_stream:
        if 'newscan' in field_names and point_dict['newscan'] > 0:
            if len(point_frame) > 0:
                point_frames.append(point_frame)
                point_frame = []
        if 'angle' in field_names and 'distant' in field_names:
            angle = point_dict.get('angle')
            distant = point_dict.get('distant')

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
            if prob > 0.5:
                measure.append([x, y])
    return np.array(measure)


def normalize_weight(weighted_points):
    total_weight = sum(w for _, w in weighted_points)
    return [[xi, w / total_weight] for xi, w in weighted_points]


def gaussian(mu, sig):
    def func(x):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    return func


def p(Z, sig):
    def given(x, func):
        return sum(gaussian(mu=0, sig=sig)(func(x, z)) for z in Z)

    return given


if __name__ == '__main__':

    ## BACKGROUND ################################################################
    ROOM_FIELD_NAMES, ROOM_POINT_STREAM = load_to_dicts('data/stream/room.csv')
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

    gx = 200
    gy = 200
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
    # cv.imshow('grid map', cv.resize(GMAP, (1000, 1000)))

    ## WALKING #################################################################
    WALKING_FIELD_NAMES, WALKING_POINT_STREAM = load_to_dicts(
        'data/stream/straight.csv')
    WALKING_FRAMES = get_frames(WALKING_POINT_STREAM, WALKING_FIELD_NAMES)[1:]
    print('length of stream\t', len(WALKING_POINT_STREAM))
    print('total number of frames\t', len(WALKING_FRAMES))

    X = np.random.uniform(low=(-3000, -3000),
                          high=(3000, 3000),
                          size=(200, 2))
    Xt = None
    speed = 1400  # mm /s
    v = [1400 / 10, 1400 / 10]

    for points in WALKING_FRAMES[:100]:
        if Xt is not None:
            X = []
            for xt in Xt:
                for xi in np.random.uniform(low=(xt - v),
                                            high=(xt + v),
                                            size=(10, 2)):
                    X.append(xi)
            X = np.array(X)

        Z = filter_measure(points, GMAP, gx, gy)
        # Z = np.array(points)
        W = np.array([p(Z, 200)(xi, find_distant) for xi in X])
        W /= sum(W)

        It = np.random.choice(range(len(X)), size=200, p=W)
        Xt = X[It]

        all_points = np.array(points)
        plt.scatter(all_points[:, 0], all_points[:, 1], s=1, c='k', alpha=0.3)
        plt.scatter(Xt[:, 0], Xt[:, 1], s=1, c='r', alpha=1)
        plt.xlim(-3000, 3000)
        plt.ylim(-3000, 3000)
        plt.show()

    # cv.waitKey(0)
