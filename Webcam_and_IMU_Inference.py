from ultralytics import YOLO
import cv2
import time
import serial
import re
import json


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

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

last_detection_time = 0
update_interval = 5  # Seconds

while True:
    success, img = cap.read()
    current_time = time.time()

    if current_time - last_detection_time > update_interval:
        if ser.inWaiting() > 0:
            serial_msg_bytes = ser.readline()
            serial_msg = serial_msg_bytes.decode(encoding='utf-8').strip()
            imu_data = parse_serial(serial_msg)
        else:
            imu_data = None

        results = model(img, stream=True)
        highest_confidence = 0
        best_detection = None

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if classNames[cls] == "person":
                    confidence = box.conf[0]
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        best_detection = box

        if best_detection:
            x1, y1, x2, y2 = best_detection.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            detection_result = {
                "label": "person",
                "bbox": [x1, y1, x2, y2],
                "prob": highest_confidence,
                "imuData": {
                    "accelerometer": imu_data['Acc'],  # Assuming the first three values are accelerometer data
                    "gyroscope": imu_data['Gyro'],  # Assuming the next three values are gyroscope data
                    "magnetometer": imu_data['Mag']  # Assuming the next three values are magnetometer data
                }
            }

            data_to_save = {
                "detection": detection_result
            }

            json_string = json.dumps(data_to_save, indent=4)
            with open("detection_result.json", "w") as outfile:
                outfile.write(json_string)

        last_detection_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
