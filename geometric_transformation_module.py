import numpy as np
import pdb
"""
--------------------------------------------
Spatial Transformer - based python module
--------------------------------------------
Adapted from:
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
    .. [3]  https://github.com/EderSantana/seya/blob/keras1/seya/layers/attention.py

    from https://github.com/oarriaga/spatial_transformer_networks/blob/master/src/spatial_transformer.py
"""

def perform_proj_transformation(image_batch, Ht, output_size, mask=None):
    ##
    ## Perform the projective warping transformation.
    ##
    aug_images = proj_transform(Ht, image_batch, output_size )
    return aug_images

def perform_aff_transformation(image_batch, Ht, output_size, mask=None):
    ##
    ## Perform the affine warping transformation.
    ##
    aug_images = aff_transform(Ht, image_batch, output_size)
    return aug_images


def _repeat(x, num_repeats):
    ones = np.ones((1, num_repeats), dtype='int32')
    x = x.reshape((-1, 1))
    x = np.matmul(x, ones)
    return x.reshape([-1])


def _interpolate(image, x, y, output_size):
    batch_size, height, width, num_channels = image.shape

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    height_float = np.float32(height)
    width_float = np.float32(width)

    output_height, output_width = output_size

    x = 0.5 * (x + 1.0) * width_float
    y = 0.5 * (y + 1.0) * height_float

    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    max_y = height - 1
    max_x = width - 1
    zero = 0

    x0 = np.clip(x0, zero, max_x)
    x1 = np.clip(x1, zero, max_x)
    y0 = np.clip(y0, zero, max_y)
    y1 = np.clip(y1, zero, max_y)

    flat_image_dimensions = width * height
    pixels_batch = np.arange(batch_size) * flat_image_dimensions
    flat_output_dimensions = output_height * output_width

    base = _repeat(pixels_batch, flat_output_dimensions)

    base_y0 = base + y0 * width
    base_y1 = base + y1 * width
    indices_a = base_y0 + x0
    indices_b = base_y1 + x0
    indices_c = base_y0 + x1
    indices_d = base_y1 + x1

    flat_image = image.reshape((-1, num_channels))
    flat_image = flat_image.astype(np.float32)
    pixel_values_a = flat_image[indices_a]
    pixel_values_b = flat_image[indices_b]
    pixel_values_c = flat_image[indices_c]
    pixel_values_d = flat_image[indices_d]

    x0 = np.float32(x0)
    x1 = np.float32(x1)
    y0 = np.float32(y0)
    y1 = np.float32(y1)

    area_a = np.expand_dims(((x1 - x) * (y1 - y)), 1)
    area_b = np.expand_dims(((x1 - x) * (y - y0)), 1)
    area_c = np.expand_dims(((x - x0) * (y1 - y)), 1)
    area_d = np.expand_dims(((x - x0) * (y - y0)), 1)

    output = np.sum([area_a * pixel_values_a,
                     area_b * pixel_values_b,
                     area_c * pixel_values_c,
                     area_d * pixel_values_d], axis=0)
    return output

def _meshgrid(height, width):
    x_linspace = np.linspace(-1., 1., width)
    y_linspace = np.linspace(-1., 1., height)
    x_coordinates, y_coordinates = np.meshgrid(x_linspace, y_linspace)
    x_coordinates = x_coordinates.flatten()
    y_coordinates = y_coordinates.flatten()
    ones = np.ones_like(x_coordinates)
    indices_grid = np.concatenate([x_coordinates, y_coordinates, ones], axis=0)
    return indices_grid

def proj_transform(proj_transformation, input_shape, output_size ):
    #
    # changed to take a projective transform
    #
    batch_size = tf.shape(input_shape)[0]
    height = tf.shape(input_shape)[1]
    width = tf.shape(input_shape)[2]
    num_channels = tf.shape(input_shape)[3]
    #
    proj_transformation = tf.reshape(proj_transformation, shape=(batch_size,3,3))
    #
    proj_transformation = tf.reshape(proj_transformation, (-1, 3, 3))
    proj_transformation = tf.cast(proj_transformation, 'float32')
    #
    width = tf.cast(width, dtype='float32')
    height = tf.cast(height, dtype='float32')
    output_height = output_size[0]
    output_width = output_size[1]
    indices_grid = _meshgrid(output_height, output_width)
    indices_grid = tf.expand_dims(indices_grid, 0)
    indices_grid = tf.reshape(indices_grid, [-1]) # flatten?
    #
    indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
    indices_grid = tf.reshape(indices_grid, (batch_size, 3, -1))
    #
    transformed_grid = tf.matmul(proj_transformation, indices_grid)
    x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
    y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
    x_s_flatten = tf.reshape(x_s, [-1])
    y_s_flatten = tf.reshape(y_s, [-1])
    #
    transformed_image = _interpolate(input_shape, x_s_flatten, y_s_flatten, output_size)
    #
    transformed_image = tf.reshape(transformed_image, shape=(batch_size, output_height, output_width, num_channels))
    #
    return transformed_image


def aff_transform(affine_transformation, input_shape, output_size):
    batch_size, height, width, num_channels = input_shape.shape

    affine_transformation = np.reshape(affine_transformation, (batch_size, 2, 3))
    affine_transformation = affine_transformation.astype('float32')

    width = float(width)
    height = float(height)
    output_height, output_width = output_size

    indices_grid = _meshgrid(output_height, output_width)
    indices_grid = np.expand_dims(indices_grid, 0)
    indices_grid = np.reshape(indices_grid, [-1])  # flatten?

    indices_grid = np.tile(indices_grid, np.stack([batch_size]))
    indices_grid = np.reshape(indices_grid, (batch_size, 3, -1))

    transformed_grid = np.matmul(affine_transformation, indices_grid)

    x_s = transformed_grid[:, 0:1, :]
    y_s = transformed_grid[:, 1:2, :]
    x_s_flatten = x_s.flatten()
    y_s_flatten = y_s.flatten()

    transformed_image = _interpolate(input_shape, x_s_flatten, y_s_flatten, output_size)

    transformed_image = np.reshape(transformed_image, (batch_size, output_height, output_width, num_channels))
    # pdb.set_trace()
    return transformed_image