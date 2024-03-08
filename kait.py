from flask import Flask
from flask import request
import json
import redis
import os


app = Flask(__name__)

@app.post("/start_imu")
def start_imu():
    loc = json.loads(request.data)
    print(loc)
    # start a python program that reads IMU data and writes to redis
    # The initial location is loc
    python('command exit code: ', os.system('python imu_integration.py'))
    return "IMU program is started"

@app.get("/get_location")
def get_location():
    r = redis.Redis(host='localhost', port=6379, db=0)
    return r.get('location')

