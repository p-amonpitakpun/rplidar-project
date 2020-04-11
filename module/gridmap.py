import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import json


def run_grid(windowName, dat, config):

    gx, gy = config['grid_size']
    xyrange = config['point_range']

    cv.namedWindow(windowName)

    while True:
        cPoints = dat.get('p')

        if cv.waitKey(10) > -1 or config.get('err') == 1:
            break
    cv.destroyAllWindows()


if __name__ == '__main__':
    cPoints = None
    with open('./out/scan.json', 'r') as scanfile:
        cPoints = json.load(scanfile)
    if cPoints is not None:
        pass
    else:
        print('error: no scanfile found')
