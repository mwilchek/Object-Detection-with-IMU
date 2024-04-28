import math
from flask import Flask, send_from_directory, current_app, request, send_file, Response
import cv2
from ultralytics import YOLO
import subprocess
import threading
import os
import datetime
import glob
import time
import atexit
import logging

# Initialize Flask app
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

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

# Directory for MP4 segments
SEGMENT_DIR = 'segments'
CAMERA_INDEX = 0 # or whatever your camera index is
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
FRAME_RATE = 30
STREAM_DELAY = 30
os.makedirs(SEGMENT_DIR, exist_ok=True)

# Cleanup old files at exit
atexit.register(lambda: [os.remove(f) for f in glob.glob(os.path.join(SEGMENT_DIR, '*.mp4'))])

def get_finalized_video_files(directory, buffer_seconds=10):
    """Retrieve list of finalized video files, excluding those modified within the last 'buffer_seconds'."""
    current_time = time.time()
    finalized_files = []
    for filepath in glob.glob(os.path.join(directory, '*.mp4')):
        mod_time = os.path.getmtime(filepath)
        if (current_time - mod_time) > buffer_seconds:  # File hasn't been modified in the last 'buffer_seconds'.
            finalized_files.append(filepath)
    return finalized_files

# Cleanup function
def cleanup_old_videos():
    files = glob.glob(os.path.join(SEGMENT_DIR, '*.mp4'))
    for f in files:
        os.remove(f)
    print("Cleaned up MP4 files.")

# Video validation function
def is_video_valid(filepath):
    ffprobe_command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=duration', '-of', 'csv=p=0', filepath]
    try:
        result = subprocess.run(ffprobe_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        duration = float(result.stdout.strip())
        return duration > 1  # Assume that a valid video should be longer than 1 second
    except subprocess.CalledProcessError:
        return False

# Filename generator
def generate_filename():
    now = datetime.datetime.now()
    filename = now.strftime('%Y%m%d%H%M%S') + '.mp4'
    return os.path.join(SEGMENT_DIR, filename)

def generate_frames():
    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

    while True:
        filename = generate_filename()
        command = [
            'ffmpeg',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', '{}x{}'.format(VIDEO_WIDTH, VIDEO_HEIGHT),
            '-r', str(FRAME_RATE),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-t', '60', # Set the segment duration to 60 seconds
            filename
        ]

        # Start FFmpeg recording process
        process = subprocess.Popen(command, stdin=subprocess.PIPE)

        start_time = time.time()
        while time.time() - start_time < STREAM_DELAY:
            # Capture frame-by-frame
            ret, frame = camera.read()
            if not ret:
                break

            resized_frame = cv2.resize(frame, (640, 480))
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

            # Write the processed frame to FFmpeg stdin
            process.stdin.write(frame.tobytes())

        # After 1 minute, stop the recording and validate the video
        process.stdin.close()
        time.sleep(1)  # Wait a second to ensure the file is written and closed properly
        process.wait()

        # Validate the video file
        if not is_video_valid(filename):
            os.remove(filename)
            print(f"Removed invalid video file: {filename}")
            continue

        print(f"New video file created: {filename}")
        #cleanup_old_videos()
        
def serve_file(path):
    """Serve the given file with support for byte range requests."""
    range_header = request.headers.get('Range', None)
    if not range_header:
        return send_from_directory(SEGMENT_DIR, os.path.basename(path))

    size = os.path.getsize(path)
    byte1, byte2 = 0, None
    m = re.search('(\d+)-(\d*)', range_header)
    g = m.groups()

    if g[0]:
        byte1 = int(g[0])
    if g[1]:
        byte2 = int(g[1])

    length = size - byte1
    if byte2 is not None:
        length = byte2 - byte1 + 1

    data = None
    with open(path, 'rb') as f:
        f.seek(byte1)
        data = f.read(length)

    rv = Response(data,
                  206,
                  mimetype='video/mp4',
                  direct_passthrough=True)
    rv.headers.add('Content-Range', 'bytes {0}-{1}/{2}'.format(byte1, byte2, size))
    return rv

@app.route('/video/')
def video():
    try:
        # Serve the latest complete and valid video file
        video_files = get_finalized_video_files(SEGMENT_DIR)
        if not video_files:
            return "No video available", 404
        
        latest_video = max(video_files, key=os.path.getctime)
        return serve_file(latest_video)
    except FileNotFoundError:
        return "Video file not found", 404

@app.route('/video/<filename>')
def serve_video(filename):
    if os.path.exists(os.path.join(SEGMENT_DIR, filename)):
        return send_from_directory(SEGMENT_DIR, filename)
    else:
        return "File not found.", 404
    
if __name__ == '__main__':
    # Start the generate_frames thread
    threading.Thread(target=generate_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
