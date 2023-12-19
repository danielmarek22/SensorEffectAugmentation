##
## code for debugging augment layers
##
#from __future__ import division
import numpy as np
import cv2
import os
import random
import math
import time
from geometric_transformation_module import perform_aff_transformation
from pix2pix_labtoRGBconv import *
import pdb
#
#     
# ---------------------------------------------------------------- #
# lens distortion augmentation functions
# ---------------------------------------------------------------- #
def aug_chromab(image, crop_h, crop_w, scale_val, tx_Rval, ty_Rval, tx_Gval, ty_Gval, tx_Bval, ty_Bval):
    # normalize image to 0-1 range, convert to float
    image_ = np.array(image) / 255.0

    # split the image into its channels
    R, G, B = np.split(image_, 3, axis=3)

    # red channel parameters
    R_alpha1 = np.ones_like(tx_Rval)
    R_alpha2 = np.zeros_like(tx_Rval)
    R_alpha3 = tx_Rval
    R_alpha4 = np.zeros_like(tx_Rval)
    R_alpha5 = np.ones_like(tx_Rval)
    R_alpha6 = ty_Rval

    # green channel parameters
    G_alpha1 = scale_val
    G_alpha2 = np.zeros_like(tx_Gval)
    G_alpha3 = tx_Gval
    G_alpha4 = np.zeros_like(tx_Rval)
    G_alpha5 = scale_val
    G_alpha6 = ty_Gval

    # blue channel parameters
    B_alpha1 = np.ones_like(tx_Rval)
    B_alpha2 = np.zeros_like(tx_Bval)
    B_alpha3 = tx_Bval
    B_alpha4 = np.zeros_like(tx_Bval)
    B_alpha5 = np.ones_like(tx_Rval)
    B_alpha6 = ty_Bval

    num_aff_params = 6
    HRt = np.stack([R_alpha1, R_alpha2, R_alpha3, R_alpha4, R_alpha5, R_alpha6], axis=1)
    HGt = np.stack([G_alpha1, G_alpha2, G_alpha3, G_alpha4, G_alpha5, G_alpha6], axis=1)
    HBt = np.stack([B_alpha1, B_alpha2, B_alpha3, B_alpha4, B_alpha5, B_alpha6], axis=1)

    # Assuming the affine transformation function is defined elsewhere in the code
    augR = perform_aff_transformation(R, HRt, (crop_h, crop_w))
    augG = perform_aff_transformation(G, HGt, (crop_h, crop_w))
    augB = perform_aff_transformation(B, HBt, (crop_h, crop_w))

    augimage = np.concatenate([augR, augG, augB], axis=3)

    # clip
    augimage = np.clip(augimage, 0.0, 1.0)
    # scale image back into 0-255 range
    augimage = np.multiply(augimage, 255.0).astype(np.uint8)
    
    # return augmented image
    return augimage

# ---------------------------------------------------------------- #
# color temperature/color balance augmentation functions
# ---------------------------------------------------------------- ##
def aug_color(image_rgb, a_transl, b_transl):
    # Convert image to CIE L*a*b* color space
    image_ = np.array(image_rgb) / 255.0
    image_lab = rgb_to_lab(image_)
    print(image_lab.shape)
    
    # Split into the 3 lab channels
    Lchan, achan, bchan = np.split(image_lab, 3, axis=3)
    
    # Apply transformations in the a and b axes
    aug_a = achan + a_transl
    aug_b = bchan + b_transl
    
    # Convert back to RGB colorspace
    auglab_ = np.squeeze(np.stack([Lchan, aug_a, aug_b], axis=3))
    augim_rgb = lab_to_rgb(auglab_)
    
    # Scale back to 0-255 range
    augimage = augim_rgb * 255.0
    
    return augimage

### ---------------------------------------------------------------- ###
### --------------- noise augmentation functions ------------------- ### 
### ---------------------------------------------------------------- ###

def aug_noise(image_rgb, batchsize, Ra_sd, Rb_si, Ga_sd, Gb_si, Ba_sd, Bb_si, im_h, im_w):
    #
    # Based upon the noise model presented in Foi et al 2009
    # Noise is modeled by the addition of a poisson (signal-dependent noise) and gaussian distribution (independent noise)
    #
    # a_sd = tf.constant(Ra_sd,shape=(batchsize,1,1,1), dtype = tf.float32)
    # b_si = tf.constant(Rb_si,shape=(batchsize,1,1,1), dtype = tf.float32) 

    ## BAYER VARIABLE DEFINITION ##
    # define matrix that captures photosite bleeding effects
    #photobleed = tf.constant(np.array([[0.95,0.04,0.01],[0.07,0.89,0.04],[0.0,0.06,0.94]]), dtype=tf.float32, shape = (1,3,3,1))
    #
    # define bayer cfa architecture
    bayer_type='GBRG'
    # define the cfa/bayer pattern (sensor locations) for each color channel
    Rcfa, Gcfa, Bcfa = return_bayer(bayer_type, im_h, im_w, batchsize) 
    # define cfa interpolation kernels
    RandB_interp = 0.25*np.array([[1,2,1],[2,4,2],[1,2,1]])
    G_interp = 0.25*np.array([[0,1,0],[1,4,1],[0,1,0]])
    Rcfa_kernel = RandB_interp[:, :, np.newaxis, np.newaxis]
    Gcfa_kernel = G_interp[:, :, np.newaxis, np.newaxis]
    Bcfa_kernel = RandB_interp[:, :, np.newaxis, np.newaxis]
    #
    # normalize images
    image_rgb_ = (np.asarray(image_rgb)/255.0).astype(float)
    #
    ## model photosite bleeding in image ##
    #image_prgb = tf.squeeze(tf.tensordot(image_rgb_, photobleed, axes=[[3],[2]]))
    #
    # split the image into its channels
    Rchan,Gchan,Bchan = np.split(image_rgb_, 3, axis=3)
    #
    ## add in realistic sensor noise to each channel ##
    Rchan_ = add_channel_noise(Rchan, Ra_sd, Rb_si, batchsize, im_h, im_w)
    Gchan_ = add_channel_noise(Gchan, Ga_sd, Gb_si, batchsize, im_h, im_w)
    Bchan_ = add_channel_noise(Bchan, Ba_sd, Bb_si, batchsize, im_h, im_w)
    #
    ## add in effects from bilinear interpolation on bayer cfa ##
    Rchan__ = bilinear_interp_cfa(Rchan_, Rcfa, Rcfa_kernel, batchsize, im_h, im_w)
    Gchan__ = bilinear_interp_cfa(Gchan_, Gcfa, Gcfa_kernel, batchsize, im_h, im_w)
    Bchan__ = bilinear_interp_cfa(Bchan_, Bcfa, Bcfa_kernel, batchsize, im_h, im_w)
    #
    # compose the noisy image:
    augnoise = np.stack([Rchan__,Gchan__,Bchan__],axis=3) 
    # scale image to 0-255
    augnoise = np.multiply(augnoise, 255.0)
    #
    augimg = augnoise
    #pdb.set_trace()
    return augimg
    #

def add_channel_noise(chan, a_sd, b_si, batchsize, im_h, im_w):
    ##
    ## determine sensor noise at each pixel using non-clipped poisson-gauss model from FOI et al 
    ##
    # if a_sd==0.0:
    #     chi=0
    #     sigdep = chan
    # else:
    #     chi = 1.0/a_sd
    #     rate = tf.maximum(chi*chan,0)
    #     sigdep = tf.random_poisson(rate, shape=[])/chi
    #     #
    chi = 1.0/a_sd
    rate = np.maximum(chi*chan,0)
    sigdep = np.random.poisson(rate) / chi
    sigindep = np.sqrt(b_si) * np.random.normal(size=(batchsize, im_h, im_w, 1), loc=0.0, scale=1.0)
    # sum the two noise sources
    chan_noise = sigdep + sigindep

    # clip the noise between 0 and 1 (baking in 0 and 255 limits)
    clip_chan_noise = np.clip(chan_noise, 0.0, 1.0)
    #
    return clip_chan_noise

def bilinear_interp_cfa(chan, cfa, cfa_kernel,batchsize, im_h, im_w):
    #
    # calculate pixel intensities based upon bayer CFA pattern
    #
    # location of pixel sensors for this color channel in the bayer array
    pix_mask = tf.equal(cfa,tf.constant(1)) 
    pix_is = chan
    pix_not = tf.zeros_like(chan)
    # get values of specific color channel sensors based upon the geometry/location of the cfa/bayer color sensors
    pix_on_cfa = tf.where(pix_mask, pix_is, pix_not)
    # use basic bilinear interpolation to solve for the noise that is in the non-pixel sensor locations
    interp_pixs = tf.nn.conv2d(pix_on_cfa, cfa_kernel, strides=[1, 1, 1, 1], padding='SAME')
    #
    #pdb.set_trace()
    return interp_pixs

def return_bayer(bayer_type, im_h, im_w, batchsize):
    #
    # generate the CFA arrays for R,G,B based upon the r pixel location:
    # 
    h = int(im_h / 2)
    w = int(im_w / 2)
    if bayer_type=='BGGR':
        # bggr
        Cr=np.array([[1,0],[0,0]])
        Cg=np.array([[0,1],[1,0]])
        Cb=np.array([[0,0],[0,1]])
        Rcfa= np.tile( Cr, (h, w))
        Gcfa= np.tile( Cg, (h, w))
        Bcfa= np.tile( Cb, (h, w))
        #
    if bayer_type=='GBRG':
        ## gbrg
        Cr2=np.array([[0,1],[0,0]])
        Cg2=np.array([[1,0],[0,1]])
        Cb2=np.array([[0,0],[1,0]])
        Rcfa= np.tile( Cr2, (h, w))
        Gcfa= np.tile( Cg2, (h, w))
        Bcfa= np.tile( Cb2, (h, w))
        #
    if bayer_type=='GRBG':
        ## grbg
        Cr3=np.array([[0,0],[1,0]])
        Cg3=np.array([[1,0],[0,1]])
        Cb3=np.array([[0,1],[0,0]])
        Rcfa= np.tile( Cr3, (h, w))
        Gcfa= np.tile( Cg3, (h, w))
        Bcfa= np.tile( Cb3, (h, w))
        #
    if bayer_type=='RGGB':
        ## rggb
        Cr4=np.array([[0,0],[0,1]])
        Cg4=np.array([[0,1],[1,0]])
        Cb4=np.array([[1,0],[0,0]])
        Rcfa= np.tile( Cr4, (h, w))
        Gcfa= np.tile( Cg4, (h, w))
        Bcfa= np.tile( Cb4, (h, w))
        #
    Rcfa= np.tile( Rcfa, (batchsize,1,1))
    Gcfa= np.tile( Gcfa, (batchsize,1,1))
    Bcfa= np.tile( Bcfa, (batchsize,1,1))
    #
    Rcfa = np.array(Rcfa, dtype=np.int32).reshape((batchsize, im_h, im_w, 1))
    Gcfa = np.array(Gcfa, dtype=np.int32).reshape((batchsize, im_h, im_w, 1))
    Bcfa = np.array(Bcfa, dtype=np.int32).reshape((batchsize, im_h, im_w, 1))
    #
    return Rcfa, Gcfa, Bcfa 

# ---------------------------------------------------------------- #
# Exposure augmentation functions
# ---------------------------------------------------------------- ##
def aug_exposure(image, delta_S, A, batchsize):
    # Ensure that image values are in the valid range [0, 255]
    image_batch = np.clip(image, 0, 255)

    # Normalize images between 0 and 1
    hin = image_batch / 255.0

    # Ensure values are in a valid range for the logarithm
    hin = np.clip(hin, 1e-10, 1.0 - 1e-10)
    # Project images into exposure space
    S = ((1.0 / hin) - 1.0)
    S = np.log(S) / -A

    # Translate images in exposure space
    Sprime = S + delta_S

    # Project augmented images back into the original image space
    Iprime = 255.0 / (1.0 + np.exp(-A * Sprime))

    # Clip values
    Iprime = np.clip(Iprime, 0.0, 255.0)

    # Scale augmented images to the 0->255 range
    hout = Iprime.astype(np.uint8)

    return hout

# ---------------------------------------------------------------- #
# Blur augmentation functions
# ---------------------------------------------------------------- ##
def aug_blur(img_inp, window_l, sig_arr, batchsize):
    # Normalize image to 0-1 range and convert to float
    image_norm = np.array(img_inp) / 255.0
    batch_list = np.split(image_norm, batchsize, axis=0)
    conv_batch_list = []

    for batch_counter, bimg in enumerate(batch_list):
        # Split channels
        Rchan, Gchan, Bchan = np.split(bimg, 3, axis=3)
        # Get the window and sigma
        wl = np.squeeze(window_l[batch_counter])
        sig = np.squeeze(sig_arr[batch_counter])
        # Get the kernel
        fgauss = gaussiankern2D(wl, sig)
        fgauss = np.squeeze(fgauss)
        # Convolution
        R_conv = cv2.filter2D(np.squeeze(Rchan), -1, fgauss.astype(np.float32))
        G_conv = cv2.filter2D(np.squeeze(Gchan), -1, fgauss.astype(np.float32))
        B_conv = cv2.filter2D(np.squeeze(Bchan), -1, fgauss.astype(np.float32))
        bimg_conv = np.stack([R_conv, G_conv, B_conv], axis=2)
        conv_batch_list.append(np.squeeze(bimg_conv))

    img_conv = np.stack(conv_batch_list, 0)
    augimage = np.squeeze(img_conv)

    # Clip
    augimage = np.clip(augimage, 0.0, 1.0)
    # Scale image back into 0-255 range
    augimage = np.multiply(augimage, 255.0)
    # Return augmented image
    return augimage

def gaussiankern2D(wl, sig):
    wx = np.arange(-wl // 2 + 1., wl // 2 + 1.)
    xx, yy = np.meshgrid(wx, wx)
    tkernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))
    tkernel_ = tkernel / np.sum(tkernel)
    expkernel_ = np.expand_dims(np.expand_dims(tkernel_, axis=2), axis=3)
    return expkernel_

### ---------------------------------------------------------------------------------------------------------------------------



