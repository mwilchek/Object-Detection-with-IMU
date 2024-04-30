import math
import cv2
import time
import os
from ultralytics import YOLO

# Open a video capture object
cap = cv2.VideoCapture(0)  # Change 0 to the index of your webcam if you have multiple cameras

# Initialize the YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get the width and height of the video frames
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use MJPEG codec
out = cv2.VideoWriter('temp.avi', fourcc, 20.0, (width, height))

# Initialize variables for time tracking
start_time = time.time()
record_duration = 10  # Record video for 10 seconds
next_record_time = start_time + record_duration

# Loop to capture video frames
while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        print("Error: Failed to capture frame.")
        break

    resized_frame = cv2.resize(frame, (width, height))  # 640, 480
    results = model(resized_frame)
    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if classNames[cls] == "person":
                detections.append(box)
    
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        confidence = math.ceil(box.conf[0] * 100) / 100
        text = f'{classNames[int(box.cls[0])]}: {confidence:.2f}'
        org = (x1, y1 - 10)
        cv2.putText(resized_frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Write the frame to the output video file
    out.write(resized_frame)

    # Check if it's time to save the video segment
    if time.time() >= next_record_time:
        out.release()  # Release the current video file
        print("Saved video segment.")
        os.system('rm output.avi')
        os.system('mv temp.avi output.avi')
        break

    # Display the frame
    #cv2.imshow('frame', resized_frame)

    # Check for key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()