import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
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
# For underwater we create a mask based on Jerlov light attenuation conditions
# instead of the classic gray fog. 
def alpha_blend(img,alpha,water_type):
    beta = 1 - alpha #weight factors
    fix_depth = 5. # fix depth in meters
    #diffuse downwelling attenuation coefficient
    #depending on Jerlov water type for B,G,R wavelengths (475,550,700)[nm]
    Kd_dict = {'II':[0.0619,0.0998,0.580],'III':[0.117,0.145,0.616],'1C':[0.134,0.145,0.616]}

    #gray_cnst = 153 This value can be used to create in-air fog (gray)
    #gray_bgr_img = np.ones(img.shape,dtype='uint8')*gray_cnst
    mask_img = np.ones(img.shape,dtype='float32')*255.
    for c in range(0,3):
        mask_img[:,:,c] = mask_img[:,:,c] * np.exp(-Kd_dict[water_type][c]*fix_depth)
    
    mask_img = mask_img.astype(dtype='uint8')
    contrast_img = cv2.addWeighted(img,alpha,mask_img, beta,0)

    return contrast_img
#--- ---#

#--- WHITE BALANCE --- #
class WhiteBalanceType(Enum):
    GRAYWORLD = 0
    SIMPLEWB = 1

def white_balance(img, wb_type, sat_thresh=None, pix_thresh=None):
    wb_img = np.zeros(img.size, dtype=img.dtype)
    if WhiteBalanceType.GRAYWORLD == wb_type:
        wb = cv2.xphoto.createGrayworldWB()
        wb.setSaturationThreshold(sat_thresh)
        wb_img = wb.balanceWhite(img)
    elif WhiteBalanceType.SIMPLEWB == wb_type:
        wb = cv2.xphoto.createSimpleWB()
        wb.setP(pix_thresh)
        wb_img = wb.balanceWhite(img)
    else:
        sys.stdout.write("NO WHITE BALANCE METHOD FOUND")
    
    return wb_img
#--- ---#

#--- IMAGE COMPRESSION --- #
class CompressionType(Enum):
    JPEG = 0
    PNG = 1
    WEBP = 2
    
def compress_img(img, comp_type, comp_val):
    comp_img = np.zeros(img.size, dtype=img.dtype)
    tmp_file = './compression_test'
    if CompressionType.JPEG == comp_type:
        cv2.imwrite((tmp_file+'.jpg'),img, [cv2.IMWRITE_JPEG_QUALITY,comp_val])
        comp_img = cv2.imread((tmp_file+'.jpg'), cv2.IMREAD_COLOR)
        os.remove((tmp_file+'.jpg'))
    elif CompressionType.PNG == comp_type:
        cv2.imwrite((tmp_file+'.png'),img, [cv2.IMWRITE_PNG_COMPRESSION,comp_val])
        comp_img = cv2.imread((tmp_file+'.png'), cv2.IMREAD_COLOR)
        os.remove((tmp_file+'.png'))
    elif CompressionType.WEBP == comp_type:
        cv2.imwrite((tmp_file+'.webp'),img, [cv2.IMWRITE_WEBP_QUALITY,comp_val])
        comp_img = cv2.imread((tmp_file+'.webp'), cv2.IMREAD_COLOR)
        os.remove((tmp_file+'.webp'))
    else:
        sys.stdout.write("NO COMPRESSION TYPE FOUND")
    return comp_img
#--- ---#