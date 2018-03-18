import os
import sys
import argparse
import numpy as np
import cv2
import json
from flask import Flask, jsonify, request

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(DIR), 'lib', 'PyOpenPose', 'build', 'PyOpenPoseLib'))
OPENPOSE_ROOT = os.path.join(os.path.dirname(DIR), 'lib', 'openpose')
import PyOpenPose as OP


parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--port", type=int, default=9989)
args = parser.parse_args()

# TODO add gpu_id, fix parameters
op = OP.OpenPose((656, 368), (368, 368), (720, 1280), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0, True, OP.OpenPose.ScaleMode.ZeroToOne, True,True)

app = Flask(__name__)

@app.route('/')
def detect():
    data = json.loads(request.data)
    bbox = data['bbox']
    print(bbox)
    # img_path = request.args.get('img_path')
    # img_path = img_path if img_path is not None else os.path.join(DIR, 'global.png')
    
    img = np.asarray(data['img'],dtype=np.uint8)
    print(img.shape)
    op.detectFace(img, np.array(bbox, dtype=np.int32).reshape((1, 4)))

    res = op.getKeypoints(op.KeypointType.FACE)[0]
    if res is None:
        return jsonify([])
    return jsonify(res.tolist())


if __name__ == "__main__":
    app.run(port=args.port)
