from dotenv import load_dotenv
import os

load_dotenv(verbose=True)

RPLIDAR_PORT = os.getenv("RPLIDAR_PORT")
if RPLIDAR_PORT is None:
    RPLIDAR_PORT = "COM3"

BAUDRATE = int(os.getenv("BAUDRATE"))
if BAUDRATE is None:
    BAUDRATE = 115200