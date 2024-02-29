from ultralytics import YOLO
import cv2
import math
import time  # Import time module

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
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

# Initialize variables to store detection results and the last update time
last_detection_time = 0
update_interval = 0  # Update interval in seconds
detections = []  # Store the last detections here

while True:
    success, img = cap.read()
    current_time = time.time()

    # Check if it's time to update detections
    if current_time - last_detection_time > update_interval:
        results = model(img, stream=True)
        detections = []  # Reset detections

        # Update detections
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])  # Get class index
                if classNames[cls] == "person":  # Check if the class name is "person"
                    detections.append(box)

        last_detection_time = current_time

    # Process and display the last detections
    for box in detections:
        # bounding box
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

        # put box in cam
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # confidence and class name
        confidence = math.ceil((box.conf[0]*100))/100
        cls = int(box.cls[0])
        text = f'{classNames[cls]}: {confidence:.2f}'

        # object details
        org = (x1, y1 - 10)  # Adjust position
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 255, 255)  # White color for text
        thickness = 2

        cv2.putText(img, text, org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
