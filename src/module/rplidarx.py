import math
import time
import numpy as np

from rplidar import RPLidar, RPLidarException, MAX_MOTOR_PWM

MOTOR_PWM = MAX_MOTOR_PWM // 2
MAX_BUFFLEN = 2000
MIN_LEN = 40


def limitPolarPoints(scanPoints):
    LOWER_DEGREE_LIMIT = 0
    UPPER_DEGREE_LIMIT = 359.99
    LOWER_DISTANT_LIMIT = 10

    new_points = []
    for point in scanPoints:
        _, degree, distance = point
        if LOWER_DEGREE_LIMIT <= degree <= UPPER_DEGREE_LIMIT and distance >= LOWER_DISTANT_LIMIT:
            new_points.append((degree, distance))
    return new_points


def cvtPolarToCartesian(polarPoint):
    degree, distance = polarPoint
    rad = np.radians(degree)
    return (distance * np.cos(rad), distance * np.sin(rad))


def run_RPLidar(port, baudrate, dat, config):

    bufflen = config.get('bufflen')
    if bufflen is None or bufflen > MAX_BUFFLEN:
        bufflen = MAX_BUFFLEN

    lidar = None
    try:
        lidar = RPLidar(port, baudrate=baudrate)
    except Exception as e:
        print(e)
    if lidar:
        try:
            lidar.clear_input()

            info = lidar.get_info()
            print(info)

            health = lidar.get_health()
            print(health)

            lidar.set_pwm(MOTOR_PWM)

            average_time = 0
            n_time = 0

            print("waiting...")
            time.sleep(5)
            t = time.time()  # start time
            # scan (quality, angle, distance)
            #degree, mm
            print("begin scan")
            for i, scan in enumerate(
                    lidar.iter_scans(max_buf_meas=bufflen, min_len=MIN_LEN)):
                start_time = time.time()
                polarPoints = limitPolarPoints(scan)
                cPoints = list(map(cvtPolarToCartesian, polarPoints))
                dat['ni'] = i
                dat['np'] = len(cPoints)
                dat['p'] = cPoints
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 10**3
                average_time = (average_time * n_time +
                                elapsed_time) / (n_time + 1)
                n_time += 1
                # et_str = str(elapsed_time)[:str(elapsed_time).find('.') + 4]
                # print(i, ': Got', len(scan), 'measurments', et_str, 'ms elasped')
        except KeyboardInterrupt:
            print("end scan")
            print("average elapsed time =", average_time, "ms")
        except Exception as e:
            print("error")
            print(e)
            if e.__str__() == 'Wrong body size':
                config['err'] = 1
        finally:
            lidar.stop()
            lidar.stop_motor()
            lidar.disconnect()
