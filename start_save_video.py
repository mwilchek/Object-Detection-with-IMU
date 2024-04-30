# Step 1: Navigate to local dir /home/user
# Step 2: Run cmd: 'http-server ./ -p 7060'
# Step 3: Navigate to repo with Object Detection with IMU
# Step 4: Run cmd: 'python start_save_video.py' 

import os
import time

# Run jetson_clocks to set the clocks to maximum performance
os.system('sudo jetson_clocks')

start_time = time.time()
record_time = 4
end_time = start_time + 10 * 60

while time.time() < end_time:
    if time.time() >= start_time + record_time:
        os.system('python save_video.py')
        start_time = time.time()
