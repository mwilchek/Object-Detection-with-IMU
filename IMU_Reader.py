import serial
import re


def parse_serial(serial_msg):
    # Regex pattern to match the formatted line from the Arduino
    pattern = (r'Mag x:([-+]?[0-9]*\.?[0-9]+) y:([-+]?[0-9]*\.?[0-9]+) z:([-+]?[0-9]*\.?[0-9]+) \| Gyro x:([-+]?['
               r'0-9]*\.?[0-9]+) y:([-+]?[0-9]*\.?[0-9]+) z:([-+]?[0-9]*\.?[0-9]+) \| Acc x:([-+]?[0-9]*\.?[0-9]+) '
               r'y:([-+]?[0-9]*\.?[0-9]+) z:([-+]?[0-9]*\.?[0-9]+)')
    match = re.search(pattern, serial_msg)
    if match:
        # Extracting all matched groups into a flat list of floats
        data = [float(match.group(i)) for i in range(1, 10)]
        return {
            "Mag": data[0:3],  # First 3 values for magnetometer
            "Gyro": data[3:6],  # Next 3 values for gyroscope
            "Acc": data[6:9]  # Last 3 values for accelerometer
        }
    return None


# Assuming '/dev/ttyACM0' is the correct port and 9600 is the baud rate
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

# Reading a single line from the serial port
serial_msg_bytes = ser.readline()
serial_msg = serial_msg_bytes.decode(encoding='utf-8').strip()

# Parsing the line to extract IMU data
imu_data = parse_serial(serial_msg)
if imu_data:
    print(f"Magnetometer: {imu_data['Mag']}")
    print(f"Gyroscope: {imu_data['Gyro']}")
    print(f"Accelerometer: {imu_data['Acc']}")
else:
    print("Failed to parse IMU data.")
