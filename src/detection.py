from module.pointprocess import run_pointprocess
from module.rplidarx import run_RPLidar
from module.gridmap import run_grid
from setting import RPLIDAR_PORT, BAUDRATE
import multiprocessing as mp
import signal


def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(
        signal))
    exit(0)


class ProcessList:
    def __init__(self):
        self._plist = []

    def add(self, target, args):
        self._plist.append(mp.Process(target=target, args=args))

    def start(self):
        for process in self._plist:
            process.start()

    def join(self):
        for process in self._plist:
            process.join()


if __name__ == '__main__':

    signal.signal(signal.SIGINT, keyboardInterruptHandler)

    n_core = mp.cpu_count()
    process_list = []

    with mp.Manager() as mng:
        plist = ProcessList()
        dat = mng.dict()
        config = mng.dict()
        config['grid_size'] = (50, 50)
        config['point_range'] = (-10000, 10000, -10000, 10000)
        config['err'] = 0

        plist.add(target=run_RPLidar,
                  args=(RPLIDAR_PORT, BAUDRATE, dat, config))
        # plist.add(target=run_pointprocess, args=(dat, config))
        # plist.add(target=run_grid, args=("grid", dat, config))

        plist.start()
        plist.join()
