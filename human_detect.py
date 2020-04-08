from module.pointprocess import run_pointprocess
from module.rplidarx import run_RPLidar
from setting import RPLIDAR_PORT, BAUDRATE
import multiprocessing as mp
import signal


def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(signal))
    exit(0)


class ProcessList:
    def __init__(self):
        self._plist = []

    def add(self, *processes):
        for process in processes:
            self._plist.append(process)

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
        plist.add(
            mp.Process(target=run_RPLidar,
                       args=(RPLIDAR_PORT, BAUDRATE, dat, config)),
            mp.Process(target=run_pointprocess, args=(dat, config)))

        plist.start()
        plist.join()