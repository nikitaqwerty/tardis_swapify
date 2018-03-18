import numpy as np
import sys
sys.path.append('/home/tardis/caffe-1.0/python')
import caffe
import os
import argparse
import cv2
from cStringIO import StringIO
from flask import Flask, jsonify, request, send_file

DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--port", type=int, default=9999)
args = parser.parse_args()
# init
caffe.set_device(0)
caffe.set_mode_gpu()
# load net
net = caffe.Net('../../data/face_seg_fcn8s_deploy.prototxt', '../../data/face_seg_fcn8s.caffemodel', caffe.TEST)


app = Flask(__name__)

@app.route('/')
def detect():
    im = np.loads(request.data)
    im = cv2.resize(im, (500, 500)) 
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)
    
    res = out.dumps()
    img_io = StringIO()
    img_io.write(res)
    img_io.seek(0)
    return send_file(img_io, mimetype='application/octet-stream')
    # if res is None:
    #     return jsonify([])
    # return res


if __name__ == "__main__":
    app.run(port=args.port)
    



