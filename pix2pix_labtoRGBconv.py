#from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function
import tensorflow as tf
import numpy as np
#import argparse
#import os
#import json
#import glob
#import random
#import collections
#import math
#
# Code from:
# https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
#

def check_image(image):
	assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
	with tf.control_dependencies([assertion]):
		image = tf.identity(image)

	if image.get_shape().ndims not in (3, 4):
		raise ValueError("image must be either 3 or 4 dimensions")
	# make the last dimension 3 so that you can unstack the colors
	shape = list(image.get_shape())
	shape[-1] = 3
	image.set_shape(shape)
	return image

# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c	
def rgb_to_lab(srgb):
    # Check image not necessary in NumPy
    srgb_pixels = srgb.reshape(-1, 3)
    
    # srgb_to_xyz
    linear_mask = (srgb_pixels <= 0.04045).astype(np.float32)
    exponential_mask = (srgb_pixels > 0.04045).astype(np.float32)
    rgb_pixels = ((srgb_pixels / 12.92) * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
    rgb_to_xyz = np.array([
        [0.412453, 0.212671, 0.019334],  # R
        [0.357580, 0.715160, 0.119193],  # G
        [0.180423, 0.072169, 0.950227],  # B
    ])
    xyz_pixels = np.dot(rgb_pixels, rgb_to_xyz)
    
    # xyz_to_cielab
    xyz_normalized_pixels = xyz_pixels * [1/0.950456, 1.0, 1/1.088754]
    epsilon = 6/29
    linear_mask = (xyz_normalized_pixels <= (epsilon**3)).astype(np.float32)
    exponential_mask = (xyz_normalized_pixels > (epsilon**3)).astype(np.float32)
    fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask
    fxfyfz_to_lab = np.array([
        [0.0, 500.0, 0.0],    # fx
        [116.0, -500.0, 200.0],  # fy
        [0.0, 0.0, -200.0],  # fz
    ])
    lab_pixels = np.dot(fxfyfz_pixels, fxfyfz_to_lab) + np.array([-16.0, 0.0, 0.0])
    
    return lab_pixels.reshape(srgb.shape)
	
def lab_to_rgb(lab):
    # Check image not necessary in NumPy
    lab_pixels = lab.reshape(-1, 3)

    # cielab_to_xyz
    lab_to_fxfyfz = np.array([
        [1/116.0, 1/116.0, 1/116.0],  # l
        [1/500.0, 0.0, 0.0],          # a
        [0.0, 0.0, -1/200.0]          # b
    ])
    fxfyfz_pixels = np.dot(lab_pixels + np.array([16.0, 0.0, 0.0]), lab_to_fxfyfz)
    epsilon = 6/29
    linear_mask = (fxfyfz_pixels <= epsilon).astype(np.float32)
    exponential_mask = (fxfyfz_pixels > epsilon).astype(np.float32)
    xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

    # denormalize for D65 white point
    xyz_pixels = xyz_pixels * np.array([0.950456, 1.0, 1.088754])

    # xyz_to_srgb
    xyz_to_rgb = np.array([
        [3.2404542, -0.9692660, 0.0556434],  # x
        [-1.5371385, 1.8760108, -0.2040259],  # y
        [-0.4985314, 0.0415560, 1.0572252]   # z
    ])
    rgb_pixels = np.dot(xyz_pixels, xyz_to_rgb)

    # avoid a slightly negative number messing up the conversion
    rgb_pixels = np.clip(rgb_pixels, 0.0, 1.0)
    linear_mask = (rgb_pixels <= 0.0031308).astype(np.float32)
    exponential_mask = (rgb_pixels > 0.0031308).astype(np.float32)
    srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

    return srgb_pixels.reshape(lab.shape)
		