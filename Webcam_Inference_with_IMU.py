import math
from ultralytics import YOLO
import cv2
import time
import serial
import re


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

# Serial port setup, update 'COM3' to your Arduino's serial port
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Model setup
model = YOLO("yolo-Weights/yolov8n.pt")

# Timing for IMU data update
last_imu_update_time = 0
imu_update_interval = 1  # seconds
last_detection_time = 0
detections = []  # Store the last detections here


while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    # Update IMU data at the specified interval
    if current_time - last_imu_update_time > imu_update_interval:
        if ser.inWaiting() > 0:
            serial_msg_bytes = ser.readline()
            serial_msg = serial_msg_bytes.decode('utf-8').strip()
            imu_data = parse_serial(serial_msg)
            last_imu_update_time = current_time

            results = model(frame, stream=True)
            detections = []  # Reset detections

            # Update detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])  # Get class index
                    if classNames[cls] == "person":  # Check if the class name is "person"
                        detections.append(box)

            last_detection_time = current_time

    # Visualize IMU data on the image
    if imu_data:
        cv2.putText(frame, f'Acc: {imu_data["Acc"]}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f'Gyro: {imu_data["Gyro"]}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f'Mag: {imu_data["Mag"]}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Process and display the last detections
    for box in detections:
        # bounding box
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

        # put box in cam
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # confidence and class name
        confidence = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])
        text = f'{classNames[cls]}: {confidence:.2f}'

        # object details
        org = (x1, y1 - 10)  # Adjust position
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 255, 255)  # White color for text
        thickness = 2

        cv2.putText(frame, text, org, font, fontScale, color, thickness)

    # Display the resulting frame
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
