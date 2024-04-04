from flask import Flask, Response, request
import cv2
import json
import redis
import os
import threading

app = Flask(__name__)


# Function to start camera stream
def start_camera_stream():
    # Open the default camera (usually the first webcam)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    # Function to generate frame by frame
    def generate_frames():
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Check if frame is empty
            if not ret:
                print("Error: Unable to read frame.")
                break

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Run camera stream on separate thread
    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    app.run(host='localhost', port=5000, db=0)


# Start camera stream on separate thread
camera_thread = threading.Thread(target=start_camera_stream)
camera_thread.daemon = True
camera_thread.start()  # http://localhost:5000/video_feed


@app.post("/start_imu")
def start_imu():
    loc = json.loads(request.data)
    print(loc)
    # start a python program that reads IMU data and writes to redis
    # The initial location is loc
    print('command exit code: ', os.system('python imu_integration.py'))
    return "IMU program is started"


@app.get("/get_location")
def get_location():
    r = redis.Redis(host='localhost', port=6379, db=0)
    return r.get('location')
