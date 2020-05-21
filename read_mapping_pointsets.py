from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from module.pointprocess import filter_polar_points, cvt_polar_to_cartesian

if __name__ == '__main__':
    pathdir = 'data/mapping/Pleum/'
    paths = glob(pathdir + '*.npy')
    for path in paths:
        print('#' * 40)
        print('read from\t', path)

        dot_index = -path[::-1].find('.')
        file_type = path[dot_index - 1:]
        file_name = path[:dot_index - 1]
        slash_index = -file_name[::-1].find('\\')
        dir_path = file_name[:slash_index]
        file_name_only = file_name[slash_index:]

        if file_type == '.npy':
            dat = np.load(path, allow_pickle=True)
            print('load\t\t', type(dat), dat.shape, 'with element',
                  type(dat[0]))
            points = []
            if len(dat) > 0 and isinstance(dat[0], dict):
                keys = dat[0].keys()
                print('with keys\t', list(keys))
                if 'angle' in keys and 'distance' in keys:
                    polars = filter_polar_points(
                        [[0, x['angle'], x['distance']] for x in dat])
                    points = np.array(
                        [cvt_polar_to_cartesian(point) for point in polars])

                    plt.scatter(points[:, 0], points[:, 1], s=1, c='k')
                    plt.xlim(-7500, 7500)
                    plt.ylim(-7500, 7500)

                    figure_path = dir_path + 'figure-' + file_name_only + '.png'
                    plt.savefig(figure_path)
                    print('saved figure\t', figure_path)

                    pointset_path = dir_path + 'pointset-' + file_name_only + '.npy'
                    np.save(pointset_path, points)
                    print('saved pointset\t', pointset_path)
        print()
