import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import json


def create_grid_map(width, height):
    return np.zeros((height, width, 3), dtype='float32')


def get_path_from_origin(point, g_x, g_y, x, y, x0, y0):
    slope = point[1] / point[0]
    path = []
    sx = int(np.sign(x))
    if x != 0 and x != -1:
        for xi in range(min(x, 0), max(x, 0) + 1):
            y1, y2 = slope * xi * g_x, slope * (xi + sx) * g_x
            yi1 = int(np.floor(y1 / g_y))
            yi2 = int(np.floor(y2 / g_y))
            for yi in range(min(yi1, yi2), max(yi1, yi2) + 1):
                if xi != x and yi != y:
                    path.append([xi + x0, yi + y0])
    else:
        for yi in range(min(y, 0), max(y, 0) + 1):
            if yi != y:
                path.append([x + x0, y + y0])
    return path


def grid_mapping(grid_map, points, g_x, g_y):
    h, w, _ = grid_map.shape
    x0, y0 = w // 2, h // 2
    for point in points:
        x, y = int(np.floor(point[0] / g_x) +
                   x0), int(np.floor(point[1] / g_y) + y0)
        path = get_path_from_origin(point, g_x, g_y, x - x0, y - y0, x0, y0)
        if x < w and y < h:
            grid_map[y, x, 1] += 1
        for x, y in path:
            if 0 <= x < w and 0 <= y < h:
                grid_map[y, x, 2] += 1
    return grid_map


def create_occupancy_grid_map(points, max_width, max_height, grid_sizes):
    grid_x, grid_y = grid_sizes
    grid_map = create_grid_map(max_width // grid_x + 1,
                               max_height // grid_y + 1)
    grid_map = grid_mapping(grid_map, points, grid_x, grid_y)
    return grid_map
