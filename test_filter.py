from tqdm import tqdm
import aedat
import complementaryfilter as cf
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

def reorder_aedat4(filename):
    decoder = aedat.Decoder(filename)
    stacked_ev = np.array(0, dtype=[('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('on', '?')])
    packets = []
    for p in tqdm(decoder):
        if 'frame' in p:
            packets.append({'t' : (p['frame']['exposure_end_t']+p['frame']['exposure_begin_t'])//2, 'p' : p})
        if 'events' in p:
            packets.append({'t' : p['events']['t'][0], 'p' : p})
    packets.sort(key=lambda i: i['t'])
    packets_f = []
    for p in packets:
        if 'frame' in p['p']:
            inds = np.where(stacked_ev['t'] <= p['t'])
            if len(inds[0]) > 0:
                packets_f.append({'t' : stacked_ev['t'][inds][0], 'p' : {'events':stacked_ev[inds]}})
            packets_f.append(p)
            stacked_ev = np.delete(stacked_ev, inds)
        if 'events' in p['p']:
            stacked_ev = np.concatenate([stacked_ev, p['p']['events']]) if len(stacked_ev.shape) > 0 else p['p']['events']
    if len(stacked_ev.shape) > 0:
        packets_f.append({'t' : stacked_ev['t'][0], 'p' : {'events':stacked_ev}})
    return packets_f

x_def = 240
y_def = 180
x = np.arange(0, x_def, 1)
x = np.tile(x, [y_def, 1])
x = np.reshape(x, x_def * y_def)
y = np.arange(0, y_def, 1)
y = np.tile(y.reshape([y_def, 1]), [1, x_def])
y = np.reshape(y, y_def * x_def)
img_d = np.zeros((y_def, x_def))
file = "data/night_run.aedat4"
th_pos = 0.3
th_neg = 0.3
alpha = 2e-6*np.pi
lam = 0.1
L1 = 10 
L2 = 250
Lmax = 255
display = True
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
        img = packet['p']['frame']['pixels'].reshape(x_def * y_def).astype(np.float32)
        img_it_ev['it'] = img
        img_it_ev['t'] = np.full(x_def * y_def, packet['p']['frame']['exposure_end_t'])
        nb_ev += img_it_ev.shape[0]
        res = cf.filterIm(img_it_ev)
        img_res[res['ev']['y'], res['ev']['x']] = res['ev']['it']
        if display:
            plt.figure(1)
            plt.clf()
            plt.subplot(1, 3, 1), plt.axis('off')
            plt.imshow(img_res / np.max(img_res)), plt.title("CF")
            plt.subplot(1, 3, 2), plt.axis('off')
            plt.imshow(np.exp(-(packet['t'] - img_ev).astype(np.float32) / 10000)), plt.title("Events")
            plt.subplot(1, 3, 3), plt.axis('off')
            ind = np.where(img_d > 0)
            img_d[ind] = np.log(img_d[ind])
            plt.imshow(img_d / np.max(img_d)), plt.title("Frames")
            plt.draw()
            plt.pause(0.1)
    if 'events' in packet['p']:
        nb_ev += packet['p']['events'].shape[0]
        res = cf.filterEv(packet['p']['events'], ex)
        img_res[res['ev']['y'], res['ev']['x']] = res['ev']['it']
        img_ev[res['ev']['y'], res['ev']['x']] = res['ev']['t']
dur = time.time() - start
print('Processing rate: {} = {} / {} ev/s'.format(nb_ev / dur, nb_ev, dur))