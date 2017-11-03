from scipy.misc import imread
from scipy.misc import imresize

import numpy as np

def make_dataset(r, act):
    x = np.zeros((0, 32*32))
    y = np.zeros((0,len(act)))
    
    
    for a in act:
        for i in r:
            
            im = imread("cropped/numbered/%s_%.4d.jpg" % (a.lower().replace(" ", "_"), i), flatten=True)
            im_f = im.flatten()
            im_f = (im_f.astype(float)-np.mean(im_f))/255.
            
            x = np.vstack((x, im_f))
            
            cur_y = np.zeros((1, len(act)))
            cur_y[0,act.index(a)] = 1
            
            y = np.vstack((y, cur_y))
    return x, y






