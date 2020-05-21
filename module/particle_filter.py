import numpy as np


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


def filter_background(points, grid_map, gx, gy, threshold, dist_threshold):
    h, w, _ = grid_map.shape
    x0, y0 = w // 2, h // 2
    background = []
    clusters = []
    cluster = []
    prev_point = None
    for point in points:
        x = int(np.floor(point[0] / gx) + x0)
        y = int(np.floor(point[1] / gy) + y0)
        if x < w and y < h:
            green = grid_map[h - y - 1, x, 1]
            red = grid_map[h - y - 1, x, 2]
            if green - red > threshold:
                background.append(point)
                if len(cluster) > 0:
                    clusters.append(cluster)
                    cluster = []
            else:
                # if prev_point is not None:
                #     if find_distant(point, prev_point) > dist_threshold and len(cluster) > 0:
                #         # print(find_distant(point, prev_point))
                #         clusters.append(cluster)
                #         cluster = []
                cluster.append(point)
            prev_point = points
    if len(cluster) > 0:
        clusters.append(cluster)
    return clusters, background


if __name__ == '__main__':

    ## BACKGROUND ############################################################
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
    cv.imshow('grid map', cv.resize(GMAP, (1000, 1000)))

    ## WALKING ###############################################################
    WALKING_FIELD_NAMES, WALKING_POINT_STREAM = load_to_dicts(
        'data/stream/straight.csv')
    WALKING_FRAMES = get_frames(WALKING_POINT_STREAM, WALKING_FIELD_NAMES)
    print('length of stream\t', len(WALKING_POINT_STREAM))
    print('total number of frames\t', len(WALKING_FRAMES))
    for points in WALKING_FRAMES[1:11]:
        clusters, room_points = filter_background(points, GMAP, gx, gy, 0,
                                                  37000)

        background_points = np.array(room_points)
        plt.scatter(background_points[:, 0],
                    background_points[:, 1],
                    s=1,
                    c='grey')
        for cluster in clusters:
            points = np.array(cluster)
            plt.scatter(points[:, 0], points[:, 1], s=1)

        plt.show()

    cv.waitKey(0)

if __name__ == '__main__':

    Ns = 100
    measure = []
    points = []

    for x in points:
        w = 0.5
