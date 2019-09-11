import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
from enum import Enum

#--- GAUSSIAN BLUR (MOTION) --- #
# Following recommendation that kernel size should be approx 6*sigma
# i.e., to cover at least 3 standard deviations in ecah direction,
# above 4 std-devs, the value of filter is very close to zero
def gaussian_blur(img,std_dev,kernel_size=None):
    if kernel_size == None:
        opt_kernel_size = int(6*std_dev)
        kernel_size= opt_kernel_size if (opt_kernel_size%2) else (opt_kernel_size-1)
    blur_img = cv2.GaussianBlur(img,(kernel_size,kernel_size),std_dev)

    return blur_img
#--- ---#

#--- BRIGHTNESS SHIFT --- #
# Apply multiplicative scaling to each channel
def brightness_shift(img,scaling_factor):
    saturated_img = img*scaling_factor
    overshoot_vals = np.argwhere(saturated_img > 255)
    overdamped_vals = np.argwhere(saturated_img < 0)
    for i,j,k in overshoot_vals:
        saturated_img[i,j,k] = 255
    for i,j in overdamped_vals:
        saturated_img[i,j,k] = 0
    
    saturated_img = saturated_img.astype(dtype='uint8')

    return saturated_img

#--- GAUSSIAN NOISE PER CHANNEL --- #
#functions to use for adding noise to RGB channels
def add_noiseToChannel(img_channel, noise):
    noisy_channel = img_channel + noise
    #minval,maxval,minloc,maxloc = cv2.minMaxLoc(noisy_channel)
    overshoot_vals = np.argwhere(noisy_channel > 255)
    overdamped_vals = np.argwhere(noisy_channel < 0)
    for i,j in overshoot_vals:
        noisy_channel[i,j] = 255
    for i,j in overdamped_vals:
        noisy_channel[i,j] = 0
    
    noisy_channel = noisy_channel.astype(dtype='uint8')
    
    return noisy_channel
    
def add_gaussNoisePerChannel(img,noise_std):
    b,g,r = cv2.split(img)
    split_channel = [b,g,r]
    noisy_img = np.zeros(img.shape,dtype='uint8')
    for x in range(0,3):
        gaussian_noise = np.zeros((img.shape[0],img.shape[1]),dtype="float")
        cv2.randn(gaussian_noise,0.0,noise_std)
        noisy_img[:,:,x] = add_noiseToChannel(split_channel[x],gaussian_noise)
    
    return noisy_img
#--- ---#

#--- CHANGE IMAGE CONTRAST --- #
def alpha_blend(img,alpha):
    beta = 1 - alpha
    gray_cnst = 153
    gray_bgr_img = np.ones(img.shape,dtype='uint8')*gray_cnst
    contrast_img = cv2.addWeighted(img,alpha,gray_bgr_img, beta,0)
    return contrast_img
#--- ---#

#--- IMAGE COMPRESSION --- #
class CompressionType(Enum):
    JPEG = 0
    PNG = 1
    WEBP = 2
    
def compress_img_fnc(img, comp_type, comp_val):
    comp_img = np.zeros(img.size, dtype=img.dtype)
    if CompressionType.JPEG == comp_type:
        cv2.imwrite('/tmp/dump/compression_test.jpg',img, [cv2.IMWRITE_JPEG_QUALITY,comp_val])
        comp_img = cv2.imread('/tmp/dump/compression_test.jpg', cv2.IMREAD_COLOR)
    elif CompressionType.PNG == comp_type:
        cv2.imwrite('/tmp/dump/compression_test.png',img, [cv2.IMWRITE_PNG_COMPRESSION,comp_val])
        comp_img = cv2.imread('/tmp/dump/compression_test.png', cv2.IMREAD_COLOR)
    elif CompressionType.WEBP == comp_type:
        cv2.imwrite('/tmp/dump/compression_test.webp',img, [cv2.IMWRITE_WEBP_QUALITY,comp_val])
        comp_img = cv2.imread('/tmp/dump/compression_test.webp', cv2.IMREAD_COLOR)
    else:
        sys.stdout.write("NO COMPRESSION TYPE FOUND")
    return comp_img
#--- ---#