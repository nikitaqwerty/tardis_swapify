import tensorflow as tf
from mtcnn.mtcnn import MTCNN
import cv2
import requests
import numpy as np
import os
import argparse
import json
import torch
from torch.autograd import Variable
from flask import Flask, jsonify, request, send_file
from multiprocessing import Process, Manager
from torch.optim.optimizer import Optimizer, required

DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--port", type=int, default=8999)
args = parser.parse_args()

app = Flask(__name__)

detector = MTCNN()

class GD(Optimizer):
    def __init__(self, params, lr=required):
        defaults = dict(lr=lr)
        super(GD, self).__init__(params, defaults)
        
    def __setstate__(self, state):
        super(GD).__setstate__(state)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)
        
        return loss

    
def findBestAffine(X, Y):
    P_X_1 = np.array([
        [1, 0 , 0],
        [0,1,0],
        [-X.mean(axis=0)[0], -X.mean(axis=0)[1], 1]
    ])
    P_Y_1 = np.array([
        [1, 0 , 0],
        [0,1,0],
        [-Y.mean(axis=0)[0], -Y.mean(axis=0)[1], 1]
    ])
    P_100 = np.array([
        [0.01, 0, 0],
        [0, 0.01, 0],
        [0,0,1],
    ])

    R_X = np.concatenate([X, np.ones((68,1))], axis=1)
    R_Y = np.concatenate([Y, np.ones((68,1))], axis=1)

    X = R_X @ P_X_1 @ P_100
    Y = R_Y @ P_Y_1 @ P_100

    T_X = Variable(torch.from_numpy(X), requires_grad=False)
    T_Y = Variable(torch.from_numpy(Y), requires_grad=False)

    A = Variable(torch.Tensor([0, 0, 0, 1]).double(), requires_grad=True)
    M_0 = Variable(torch.Tensor([0]).double(), requires_grad=False)
    M_1 = Variable(torch.Tensor([1]).double(), requires_grad=False)

    optimizer = GD([A,], lr=0.01)

    for _ in range(20):
        Q = torch.stack([
            torch.cat([A[3] * torch.cos(A[0]), - A[3] * torch.sin(A[0]), M_0]),
            torch.cat([A[3] * torch.sin(A[0]), A[3] * torch.cos(A[0]), M_0]),
            torch.cat([A[1], A[2], M_1])], dim=0)

        optimizer.zero_grad()
        loss_vec = T_X @ Q - T_Y
        loss = (loss_vec ** 2).sum()
#         print(loss.data[0])
        loss.backward()
        optimizer.step()

    Q_n = Q.data.numpy()

    F = P_X_1 @ P_100 @ Q_n @ np.linalg.inv(P_Y_1 @ P_100)
    # print(R_X @ F - R_Y)
    F = F.T[:2, :].astype(np.float64)
    return F

def faceswap(im1, im2,output1,output2):
    mask1 = np.array(output1['seg_mask'])
    mask2 = np.array(output2['seg_mask'])
    imm1 = im1.copy()
    imm1[mask1 == 0] = (0,0,0)
    imm2 = im2.copy()
    imm2[mask2 == 0] = (0,0,0)
    # plt.imshow(imm2)

    X = np.array(output2['landmarks'])[:68, :2]
    Y = np.array(output1['landmarks'])[:68, :2]


    F = findBestAffine(X, Y)
    M2 = np.stack([mask2, mask2, mask2], axis=2).astype(np.uint8)
    L = cv2.warpAffine(M2, F, (imm1.shape[:2][1], imm1.shape[:2][0]))
    LL = cv2.warpAffine(im2, F, (imm1.shape[:2][1], imm1.shape[:2][0]))

    face = LL.astype(np.uint8)
    mask = L.astype(np.uint8)
    mask[mask>0] = 255
    img = im1

    print(mask.shape)
    print(face.shape)
    print(img.shape)

    m = cv2.moments(mask[:, :, 0])

    c1 = m['m10'] / m['m00']
    c0 = m['m01'] / m['m00']

    center = (np.int32(c1), np.int32(c0))

    q = cv2.seamlessClone(face, img, mask, center, cv2.NORMAL_CLONE)
    return q

def get_landmarks(bbox,img):
    payload = {'bbox':bbox, 'img':img.tolist()}
    data = json.dumps(payload)
    output = requests.get('http://127.0.0.1:9989/',data=data)
    landmarks = np.array(output.json()[0]).reshape((70,3))
    return landmarks
    
def get_segmask(bbox,img):
    margin = 50
    if (bbox[1]<margin) or (bbox[0]<margin) or ((bbox[1]+bbox[3]+margin) > img.shape[0]) or ((bbox[0]+bbox[2]+margin) > img.shape[1]):
        margin = 0
    
    print('Margin =',margin)
    
    img_bboxed = img[bbox[1]-margin:bbox[1]+bbox[3]+margin,bbox[0]-margin:bbox[0]+bbox[2]+margin,:]
    
    desired_size = np.max(img_bboxed.shape)
    old_size = img_bboxed.shape[:2] # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    print('bboxedshape',img_bboxed.shape)
    print('new_size',new_size)

    img_bboxed = cv2.resize(img_bboxed, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(img_bboxed, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    
    output = requests.get('http://127.0.0.1:9999/',data=np.array(new_im).dumps())
    seg_mask = np.loads(output.content, encoding='latin1').astype(np.uint8)
    seg_mask = seg_mask[left:seg_mask.shape[0]-right,top:seg_mask.shape[1] - bottom]
    seg_mask = cv2.resize(seg_mask,(old_size[1],old_size[0]))
    seg_mask = cv2.copyMakeBorder(seg_mask,bbox[1]-margin,img.shape[0] - (bbox[1]+bbox[3]+margin),bbox[0]-margin,
                                   img.shape[1] - (bbox[0]+bbox[2]+margin), cv2.BORDER_CONSTANT,value=color)
    return seg_mask

def process_img(img):
    detected = detector.detect_faces(img)
    bbox = detected[0]['box']
    print(img.shape)
    print(bbox)
    landmarks = get_landmarks(bbox,img)
    seg_mask = get_segmask(bbox,img)
    
#     manager = Manager()
#     res_dict = manager.dict()
    
#     p1 = Process(target=get_landmarks,args=(bbox,img,res_dict))
#     p1.start()
#     p2 = Process(target=get_segmask,args=(bbox,img,res_dict))
#     p2.start()
#     p1.join()
#     p2.join()
    # landmarks = res_dict['landmarks']
    # seg_mask = res_dict['seg_mask']
    return landmarks,seg_mask

@app.route('/points')
def detect():
    img = np.loads(request.data, encoding='latin1')
    landmarks,seg_mask = process_img(img)
    payload = {'landmarks':landmarks.tolist(), 'seg_mask':seg_mask.tolist()}
    if payload is None:
        return jsonify([])
    return jsonify(payload)

@app.route('/swap')
def swap():
    print('GOT SWAP REQ')
    print(type(request.data))
    data = json.loads(str(request.data,'utf-8'))
    print(type(data))
    print(data.keys())

    
    print('LOADED DATA')
    print(type(data['img_from']))
    img_from = np.asarray(data['img_from'],dtype=np.uint8)
    print('FROM SHAPE',img_from.shape)
    img_to = np.asarray(data['img_to'],dtype=np.uint8)
    print('TO SHAPE',img_to.shape)
    
    landmarks,seg_mask = process_img(img_from)
    output2 = {'landmarks': landmarks,'seg_mask': seg_mask}
    landmarks,seg_mask = process_img(img_to)
    output1 = {'landmarks': landmarks,'seg_mask': seg_mask}
    
    swapped = faceswap(img_to,img_from,output1,output2)

#     o = cv2.imread(os.path.join(DIR,'i.jpg'))
    
#     o = cv2.resize(o, (img_to.shape[1], img_to.shape[0]))
    # print('RESIZED',o.shape)
    return jsonify(swapped.tolist())


if __name__ == "__main__":
    app.run(port=args.port)
    
