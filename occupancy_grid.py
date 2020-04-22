import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from module.data import load_to_dicts
from module.rplidarx import cvtPolarToCartesian


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


def get_grid_map(gx, gy, frame_size):
    frame_width, frame_height = frame_size
    map_size = (frame_height // gy + 1, frame_width // gx + 1, 3)
    grid_map = np.zeros(map_size, dtype='float32')
    return grid_map


def get_path_from_origin(point, gx, gy, x, y, x0, y0):
    slope = point[1] / point[0]
    path = []
    sx = int(np.sign(x))
    if x != 0 and x != -1:
        for xi in range(min(x, 0), max(x, 0) + 1):
            y1, y2 = slope * xi * gx, slope * (xi + sx) * gx
            yi1 = int(np.floor(y1 / gy))
            yi2 = int(np.floor(y2 / gy))
            for yi in range(min(yi1, yi2), max(yi1, yi2) + 1):
                if xi != x and yi != y:
                    path.append([xi + x0, yi + y0])
    else:
        for yi in range(min(y, 0), max(y, 0) + 1):
            if yi != y:
                path.append([x + x0, y + y0])
    return path


def grid_mapping(grid_map, points, gx, gy):
    h, w, _ = grid_map.shape
    x0, y0 = w // 2, h // 2
    for point in points:
        x, y = int(np.floor(point[0] / gx) +
                   x0), int(np.floor(point[1] / gy) + y0)
        path = get_path_from_origin(point, gx, gy, x - x0, y - y0, x0, y0)
        if x < w and y < h:
            grid_map[h - y - 1, x, 1] += 0.9
        for x, y in path:
            if 0 <= x < w and 0 <= y < h:
                grid_map[h - y - 1, x, 2] += 0.7
    return grid_map

def filter_background(points, grid_map, gx, gy, threshold):
    h, w, _ = grid_map.shape
    x0, y0 = w // 2, h // 2
    clusters = []
    cluster = []
    for point in points:
        x= int(np.floor(point[0] / gx) + x0)
        y = int(np.floor(point[1] / gy) + y0)
        if x < w and y < h:
            green = grid_map[h - y - 1, x, 1]
            red = grid_map[h - y - 1, x, 2]

            # print(green - red)

            if green - red > threshold and len(cluster) > 0:
                clusters.append(cluster)
                cluster = []
            else:
                cluster.append(point)
    if len(cluster) > 0:
        clusters.append(cluster)
    return clusters


if __name__ == '__main__':
    ROOM_FIELD_NAMES, ROOM_POINT_STREAM = load_to_dicts('data/stream/room.csv')
    ROOM_FRAMES = get_frames(ROOM_POINT_STREAM, ROOM_FIELD_NAMES)
    print('length of stream\t', len(ROOM_POINT_STREAM))
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

    gx = 100
    gy = 100
    GMAP = get_grid_map(gx, gy, (10000, 10000))
    N_FRAME = 10
    for points in ROOM_FRAMES[0:N_FRAME]:
        GMAP = grid_mapping(GMAP, points, gx, gy)
    GMAP = GMAP / (N_FRAME if N_FRAME > 0 else len(ROOM_FRAMES))
    print('shape\t\t\t', GMAP.shape)
    print('number of frames\t', N_FRAME if N_FRAME > 0 else len(ROOM_FRAMES))
    GMAP = cv.resize(GMAP, (1000, 1000))
    cv.imshow('grid map', GMAP)

    WALKING_FIELD_NAMES, WALKING_POINT_STREAM = load_to_dicts(
        'data/stream/straight.csv')
    WALKING_FRAMES = get_frames(WALKING_POINT_STREAM, WALKING_FIELD_NAMES)
    print('length of stream\t', len(WALKING_POINT_STREAM))
    print('total number of frames\t', len(WALKING_FRAMES))

    # print(np.array([len(frame) for frame in WALKING_FRAMES]))

    # print(ROOM_POINTS)
    for points in WALKING_FRAMES[: 2]:
        clusters = filter_background(points, GMAP, gx, gy, -20)
        # print("number of clusters ", len(points))
        # print(np.array([len(cluster) for cluster in clusters]))

        WALKING_POINTS = np.array(points)
        plt.scatter(ROOM_POINTS[:, 0], ROOM_POINTS[:, 1], s=1, c='k')
        plt.scatter(WALKING_POINTS[:, 0], WALKING_POINTS[:, 1], s=1, c='r')
        for cluster in clusters:
            if len(cluster) > 10:
                CLUSTER_POINTS = np.array(cluster)
                plt.scatter(CLUSTER_POINTS[:, 0], CLUSTER_POINTS[:, 1], s=1, c='g')
        plt.show()

    cv.waitKey(0)
