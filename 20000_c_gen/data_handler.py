import numpy as np
import cv2
import skvideo.io
import bouncing_balls as b
from matplotlib import pyplot as plt
import db_wt as dbwt

# the codec character code
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')

def generate_bouncing_ball_sample(batch_size, seq_length, shape, num_balls):
    """
    Creates sequences of bouncing balls.
    Args:
        batch_size: integer, number of sequences to generate
        seq_length: number of frames to generate for each sequence
        shape: [m, n, k] list where m and n are the frame height and width,
               and k is the number of channels ... *** note: m must = n!
        num_balls: number of balls to generate
    """
    dat = np.zeros((batch_size, seq_length, shape[0], shape[1], 3))
    for i in xrange(batch_size):
        dat[i, :, :, :, :] = b.bounce_vec(shape[0], num_balls, seq_length)
    if shape[2] == 1:
        rgb_dat = np.zeros((batch_size, seq_length, shape[0], shape[1], 1))
        rgb_dat[:, :, :, :, 0] = np.dot(dat[..., :3], [0.299, 0.587, 0.114]) #make grayscale
        return rgb_dat
    return dat

def generate_wavelet_sample(batch_size, seq_length, shape, num_balls, k, m, band):
    # *** NOTE: SEQUENCE LENGTH MUST BE EVEN!!!
    data = generate_bouncing_ball_sample(batch_size, 18, shape, num_balls)
    d_mov = np.zeros((data.shape[0],) + (data.shape[1]/2 + k,) + data.shape[2:])
    c_mov = np.zeros((data.shape[0],) + (data.shape[1]/2 + k,) + data.shape[2:])  
    
    for bn in range(data.shape[0]):
        for i in range(data.shape[2]):
            for j in range(data.shape[3]):
                for c in range(data.shape[-1]):
                    coarse, detail = dbwt.forward(data[bn, :, i, j, c], m, k)
                    c_mov[bn, :, i, j, c], d_mov[bn, :, i, j, c] = coarse, detail[0]
    
    if band == 'coarse':
        return c_mov
    else:
        return d_mov

"""
# *** NOTE: SEQUENCE LENGTH MUST BE EVEN!!!

# here's a sample

# add wavelet functions here!
batch_size = 2
seq_length = 300
shape = [32, 32, 1]
num_balls = 2

# generate some data
data = generate_bouncing_ball_sample(batch_size, seq_length, shape, num_balls)

# db wavelet transform parameters
k = 5
m = 1

d_mov = np.zeros((data.shape[0],) + (data.shape[1]/2 + k,) + data.shape[2:])
c_mov = np.zeros((data.shape[0],) + (data.shape[1]/2 + k,) + data.shape[2:])  

for bn in range(data.shape[0]):
    for i in range(data.shape[2]):
        for j in range(data.shape[3]):
            for c in range(data.shape[-1]):
                coarse, detail = dbwt.forward(data[bn, :, i, j, c], m, k)
                c_mov[bn, :, i, j, c], d_mov[bn, :, i, j, c] = coarse, detail[0]


# if we have a 1 level transform, then we can simply concat d and c to form vid
if m == 1:
    #fourcc = cv2.cv.FOURCC('m', 'p', '4', 'v')
    #vid = cv2.VideoWriter()
    #success = vid.open('movie.mov', fourcc, 4, (360, 360), True)
    video = np.zeros([d_mov.shape[1], 360, 360, 1])
    for i in range(d_mov.shape[1]):
        # detail
        x_1_r = np.uint8(np.maximum(d_mov[0, i, :, :, :], 0) * 255)
        d_im = cv2.resize(x_1_r, (180, 180))
        # coarse
        x_1_r = np.uint8(np.maximum(c_mov[0, i, :, :, :], 0) * 255)
        c_im = cv2.resize(x_1_r, (180, 180))
        # concatenate
        b0 = np.hstack((c_im, d_im))
        
        # detail
        x_1_r = np.uint8(np.maximum(d_mov[1, i, :, :, :], 0) * 255)
        d_im = cv2.resize(x_1_r, (180, 180))
        # coarse
        x_1_r = np.uint8(np.maximum(c_mov[1, i, :, :, :], 0) * 255)
        c_im = cv2.resize(x_1_r, (180, 180))
        # concatenate
        b1 = np.hstack((c_im, d_im))
        
        final = np.vstack((b0, b1))
        
        video[i, :, :, 0] = final
        
    skvideo.io.vwrite('gen_video.mp4', video, backend='libav')
"""