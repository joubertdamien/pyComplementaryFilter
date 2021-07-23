import aedat
import complementaryfilter as cf
import numpy as np
import sys
import time
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

def reorder_aedat4(filename):
    decoder = aedat.Decoder(filename)
    packets = []
    for packet in tqdm(decoder):
        if 'frame' in packet:
            packets.append({'t' : packet['frame']['exposure_end_t'], 'p' : packet})
        if 'events' in packet:
            packets.append({'t' : packet['events']['t'][0], 'p' : packet})
    packets.sort(key=lambda i: i['t'])
    return packets

x_def = 240
y_def = 180
x = np.arange(0, x_def, 1)
x = np.tile(x, [y_def, 1])
x = np.reshape(x, x_def * y_def)
y = np.arange(0, y_def, 1)
y = np.tile(y.reshape([y_def, 1]), [1, x_def])
y = np.reshape(y, y_def * x_def)

file = "data/night_run.aedat4"
th_pos = 0.4
th_neg = 0.4
alpha = 0.00001*np.pi
lam = 0.1
L1 = 10 
L2 = 250
Lmax = 255
display = False
cf.init(x_def, y_def, th_pos, th_neg, alpha, lam, L1, L2, Lmax)
img_it_ev = np.zeros(x_def * y_def, dtype=[('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('it', 'f4')])
ex = np.zeros(1, dtype=[('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('it', 'f4')])
img_it_ev['x'] = x
img_it_ev['y'] = y
img_res = np.zeros((y_def, x_def), dtype=np.float)
img_ev = np.zeros((y_def, x_def), dtype=np.uint64)
packets = reorder_aedat4(file)
nb_ev = 0
start = time.time() 
for packet in packets:
    if 'frame' in packet['p']:
        img_d = np.array(packet['p']['frame']['pixels'])
        img = packet['p']['frame']['pixels'].reshape(x_def * y_def)
        ind = np.where(img > 0)
        img[ind] = np.log(img[ind])
        img_it_ev['it'] = img
        img_it_ev['t'] = np.full(x_def * y_def, packet['p']['frame']['exposure_end_t'])
        nb_ev += img_it_ev.shape[0]
        res = cf.filterIm(img_it_ev)
        img_res[res['ev']['y'], res['ev']['x']] = res['ev']['it']
        if display:
            plt.figure(1)
            plt.clf()
            plt.subplot(1, 3, 1)
            plt.imshow(img_res / np.max(img_res)), plt.title("CF")
            plt.subplot(1, 3, 2)
            plt.imshow(np.exp(-(packet['p']['frame']['t'] - img_ev).astype(np.float32) / 100000)), plt.title("Events")
            plt.subplot(1, 3, 3)
            plt.imshow(img_d), plt.title("Frames")
            plt.draw()
            plt.pause(0.01)
    if 'events' in packet['p']:
        nb_ev += packet['p']['events'].shape[0]
        res = cf.filterEv(packet['p']['events'], ex)
        img_res[res['ev']['y'], res['ev']['x']] = res['ev']['it']
        img_ev[res['ev']['y'], res['ev']['x']] = res['ev']['t']
dur = time.time() - start
print('Processing rate: {} = {} / {} ev/s'.format(nb_ev / dur, nb_ev, dur))
       