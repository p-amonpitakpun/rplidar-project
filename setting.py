from dotenv import load_dotenv
import os

load_dotenv(verbose=True)

RPLIDAR_PORT = os.getenv("RPLIDAR_PORT")
BAUDRATE = int(os.getenv("BAUDRATE"))