from glob import glob
from time import time
import matplotlib.pyplot as plt
import numpy as np
from module.gridmap import create_occupancy_grid_map
from module.icp import tr_icp, icp, closest_points
from module.particle_filter import filter_background

if __name__ == '__main__':
    pathdir = 'data/mapping/Pleum/'
    pointsets = []
    paths = glob(pathdir + 'pointset-*.npy')
    for path in paths:
        pointsets.append(np.load(path))

    room_pointsets = []
    for points in pointsets:
        print('load', type(points), points.shape)
        n_point = 1000
        m_point = 2000
        grid_map = create_occupancy_grid_map(points[:n_point], 7500, 7500,
                                             (50, 50))
        # cv2.imshow('grid map', cv2.resize(grid_map, (1000, 1000)))
        _, room_points_list = filter_background(points[:m_point], grid_map, 50,
                                                50, 0, 0)
        room_points = np.array(room_points_list)
        # plt.scatter(room_points[:, 0], room_points[:, 1], s=1, c='k', alpha=0.3)
        # plt.xlim(-7500, 7500)
        # plt.ylim(-7500, 7500)
        # plt.show()
        room_pointsets.append(room_points)
    print()

    for pointset, next_pointset in zip(room_pointsets[:-1],
                                       room_pointsets[1:]):
        n_point = 500
        # np.random.shuffle(pointset)
        # np.random.shuffle(next_pointset)
        ref_points = pointset[:n_point]
        icp_points = next_pointset[:n_point]
        # X = closest_points(icp_points, ref_points)
        # err = np.sum(
        #     np.sqrt(np.sum(
        #         (X[:] - icp_points[:])**2, axis=1))) / n_point
        # print('before icp err\t', err)
        print('compute icp of\t', ref_points.shape, 'and', icp_points.shape)
        start_time = time()

        # compute ICP
        # R, T = tr_icp(ref_points, icp_points, N=20, N_iter=20)
        R, T = icp(ref_points, icp_points, N_iter=40)

        print('compute time\t', time() - start_time, 's')
        print('\nr=\n', R)
        print('\nt=\n', T)
        target_points = np.matmul(R, icp_points.T).T + T
        print()

        # plot
        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
        ax[0].scatter(ref_points[:, 0],
                      ref_points[:, 1],
                      s=1,
                      c='b',
                      alpha=0.3)
        ax[0].scatter(icp_points[:, 0],
                      icp_points[:, 1],
                      s=1,
                      c='r',
                      alpha=0.3)
        ax[1].scatter(ref_points[:, 0],
                      ref_points[:, 1],
                      s=1,
                      c='b',
                      alpha=0.3)
        ax[1].scatter(target_points[:, 0],
                      target_points[:, 1],
                      s=1,
                      c='g',
                      alpha=0.3)
        for a in ax:
            a.set_xlim(-7500, 7500)
            a.set_ylim(-7500, 7500)
        plt.show()
