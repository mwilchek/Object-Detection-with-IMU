import time
import serial
import re
import torch
import pypose as pp
import redis
import json


def parse_serial(serial_msg):
    pattern = (r'Mag x:([-+]?[0-9]*\.?[0-9]+) y:([-+]?[0-9]*\.?[0-9]+) z:([-+]?[0-9]*\.?[0-9]+) \| Gyro x:([-+]?['
               r'0-9]*\.?[0-9]+) y:([-+]?[0-9]*\.?[0-9]+) z:([-+]?[0-9]*\.?[0-9]+) \| Acc x:([-+]?[0-9]*\.?[0-9]+) '
               r'y:([-+]?[0-9]*\.?[0-9]+) z:([-+]?[0-9]*\.?[0-9]+)')
    match = re.search(pattern, serial_msg)
    if match:
        data = [float(match.group(i)) for i in range(1, 10)]
        return {
            "Mag": data[0:3],
            "Gyro": data[3:6],
            "Acc": data[6:9]
        }
    return None


# Serial port setup, update 'COM3' to your Arduino's serial port
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)


# Timing for IMU data update
last_imu_update_time = 0
imu_update_interval = 0.01  # seconds


integrator = pp.module.IMUPreintegrator()
r = redis.Redis(host='localhost', port=6379, db=0)

ts_dt = torch.tensor([1 / 99.84])
while True:
    current_time = time.time()
    # Update IMU data at the specified interval
    dt = current_time - last_imu_update_time
    if dt > imu_update_interval:
        if ser.in_waiting > 0:
            serial_msg_bytes = ser.readline()
            serial_msg = serial_msg_bytes.decode('utf-8').strip()
            imu_data = parse_serial(serial_msg)
            last_imu_update_time = current_time
            if imu_data is None:
                continue
            
            ts_acc = torch.tensor(imu_data['Acc'])
            ts_gyro = torch.tensor(imu_data['Gyro'])
            
            integrator(dt=ts_dt, gyro=ts_gyro, acc=ts_acc)
            print(integrator.pos)
            loc = {'x': integrator.pos[0, 0, 0], 'y': integrator.pos[0, 0, 1], 'z': integrator.pos[0, 0, 2]}
            r.set('location', json.dumps(loc))

ser.close()
