import numpy as np import cv2 import copy import matplotlib.pyplot as plt import json import requests to = cv2.imread('to1.jpg') to = cv2.cvtColor(to, 
cv2.COLOR_BGR2RGB) fr = cv2.imread('from.jpg') fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) output = requests.get('http://13.90.200.154:9213/points', data=to.dumps()) 
topnts = np.array(output.json()['landmarks']) tosgm = np.array(output.json()['seg_mask'])/15 output = requests.get('http://13.90.200.154:9213/points', data=fr.dumps()) 
frpnts = np.array(output.json()['landmarks']) frsgm = np.array(output.json()['seg_mask'])/15
# cutting face from source image
mask = np.zeros(fr.shape[:2],np.uint8) bgdModel = np.zeros((1,65),np.float64) fgdModel = np.zeros((1,65),np.float64) w = int((frpnts[16,0] - frpnts[0,0]) * 1.5) h = 
int((frpnts[8,1] - frpnts[24,1]) * 2) x = int(frpnts[0,0] - w/6) y = int(frpnts[24,1] - h/2) rect = (x,y,w,h) 
cv2.grabCut(fr,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT) r_channel, g_channel, b_channel = cv2.split(fr) a_channel = 
np.where((mask==3)|(mask==1)|(frsgm==1), 255, 0).astype('uint8') fralpha = cv2.merge((r_channel, g_channel, b_channel, a_channel)) mask2 = 
np.where((mask==3)|(mask==1)|(frsgm==1), 1, 0).astype('uint8') frcut = fralpha*mask2[:,:,np.newaxis] frcut = frcut[y:(y+h+20), x:(x+w)] a_channel = 
a_channel[y:(y+h+20), x:(x+w)] nx, ny = (frcut.shape[1], 20) xx = np.linspace(1, 0, nx) yy = np.linspace(1, 0, ny) xv, yv = np.meshgrid(xx, yy) a_channel2 = 
copy.deepcopy(a_channel) a_channel2[(a_channel2.shape[0]-20):a_channel2.shape[0], :] = yv*255 a_channel2[a_channel==0] = 0 frcut[:,:,3] = a_channel2 sbstr = 
np.array([x, y, 0]) sbstr = np.outer(sbstr, np.ones(70)) frpntscut = frpnts - sbstr.transpose()
# cutting face from destination image
mask = np.zeros(to.shape[:2],np.uint8) bgdModel = np.zeros((1,65),np.float64) fgdModel = np.zeros((1,65),np.float64) w = int((topnts[16,0] - topnts[0,0]) * 1.5) h = 
int((topnts[8,1] - topnts[24,1]) * 2) x = int(topnts[0,0] - w/6) y = int(topnts[24,1] - h/2) rect = (x,y,w,h) 
cv2.grabCut(to,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT) kernel = np.ones((7,7),np.uint8) mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8') tocut = 
to*mask2[:,:,np.newaxis] mask2 = cv2.dilate(mask2, kernel, 1) mask2 = cv2.dilate(mask2, kernel, 1) toinp = cv2.inpaint(to,mask2, 9, cv2.INPAINT_NS)
# resizing frcut
k1 = (topnts[16,0] - topnts[0,0])/(frpnts[16,0] - frpnts[0,0]) k2 = (topnts[24,1] - topnts[8,1])/(frpnts[24,1] - frpnts[8,1]) k = (k1+k2)/2 frcutr = cv2.resize(frcut, 
(int(frcut.shape[1]*k), int(frcut.shape[0]*k))) frpntscutk = frpntscut*k
# adding frcut to toinp
x_offset = int(topnts[8,0] - frpntscutk[8,0]) y_offset = int(topnts[8,1] - frpntscutk[8,1]) if y_offset<0:
    frcutr = frcutr[(-y_offset):frcutr.shape[0],:]
    y1, y2 = 0, frcutr.shape[0] else:
    y1, y2 = y_offset, y_offset + frcutr.shape[0] x1, x2 = x_offset, x_offset + frcutr.shape[1] alpha_s = frcutr[:, :, 3] / 255.0 alpha_l = 1.0 - alpha_s res = 
copy.deepcopy(toinp) for c in range(0, 3):
    res[y1:y2, x1:x2, c] = (alpha_s * frcutr[:, :, c] +
                              alpha_l * toinp[y1:y2, x1:x2, c]) res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
cv2.imwrite("res1.png", res)
