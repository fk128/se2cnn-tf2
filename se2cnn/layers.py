import tensorflow as tf
from tensorflow.keras import layers
from .layer_ops import rotate_lifting_kernels, rotate_gconv_kernels, spatial_max_pool
import numpy as np


class LiftingLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, n_theta=12, strides=(1, 1, 1, 1),
                 activation=None,
                 periodicity=2 * np.pi,
                 disk_mask=True,
                 padding='VALID',
                 use_bias=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel = kernel
        self.n_theta = n_theta
        self.strides = strides
        self.activation = tf.keras.activations.get(activation)
        self.periodicity = periodicity
        self.disk_mask = disk_mask
        self.padding = padding.upper()
        self.use_bias = use_bias

    def build(self, input_shape):
        initializer = tf.keras.initializers.GlorotNormal(seed=None)
        self.w = self.add_weight(
            shape=(self.kernel, self.kernel, input_shape[-1], self.filters),
            initializer=initializer,
            trainable=True,
            dtype="float32"
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(1, 1, 1, 1, self.filters), initializer="random_normal", trainable=True
            )

    def rotate_kernel(self):
        # Preparation for group convolutions
        # Precompute a rotated stack of kernels
        kernel_stack = rotate_lifting_kernels(
            self.w, self.n_theta, periodicity=self.periodicity,
            diskMask=self.disk_mask)

        # print("Z2-SE2N ROTATED KERNEL SET SHAPE:",
        #       kernel_stack.get_shape())  # Debug

        # Format the kernel stack as a 2D kernel stack (merging the rotation and
        # channelsOUT axis)
        kernels_as_if_2D = tf.cast(tf.transpose(
            kernel_stack, [1, 2, 3, 0, 4]), dtype="float32")
        kernelSizeH, kernelSizeW, channelsIN, channelsOUT = map(
            int, self.w.shape)
        kernels_as_if_2D = tf.reshape(
            kernels_as_if_2D, [kernelSizeH, kernelSizeW, channelsIN, self.n_theta * channelsOUT])
        return kernels_as_if_2D, kernel_stack

    def call(self, inputs, **kwargs):

        kernels_as_if_2D, _ = self.rotate_kernel()

        # Perform the 2D convolution
        output = tf.nn.conv2d(
            input=inputs,
            filters=kernels_as_if_2D,
            strides=self.strides,
            padding=self.padding)

        # Reshape to an SE2 image (split the orientation and channelsOUT axis)
        # Note: the batch size is unknown, hence this dimension needs to be
        # obtained using the tensorflow function tf.shape, for the other
        # dimensions we keep using tensor.shape since this allows us to keep track
        # of the actual shapes (otherwise the shapes get convert to
        # "Dimensions(None)").
        output = tf.reshape(
            output, [tf.shape(output)[0],
                     int(output.shape[1]), int(output.shape[2]),
                     self.n_theta, self.filters])
        # print("OUTPUT SE2N ACTIVATIONS SHAPE:", output.get_shape())  # Debug
        if self.use_bias:
            output += self.b

        return self.activation(output)


class SE2Layer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, n_theta=12, activation=None,
                 periodicity=2 * np.pi,
                 disk_mask=True,
                 strides=(1, 1, 1, 1),
                 padding='VALID',
                 use_bias=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.n_theta = n_theta
        self.activation = tf.keras.activations.get(activation)
        self.periodicity = periodicity
        self.disk_mask = disk_mask
        self.padding = padding.upper()
        self.use_bias = use_bias

    def build(self, input_shape):
        initializer = tf.keras.initializers.GlorotNormal(seed=None)
        self.w = self.add_weight(
            shape=(self.kernel, self.kernel, self.n_theta,
                   input_shape[-1], self.filters),
            initializer=initializer,
            trainable=True,
            dtype="float32"
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(1, 1, 1, 1, self.filters), initializer="random_normal", trainable=True
            )

    def rotate_kernel(self):

        # Preparation for group convolutions
        # Precompute a rotated stack of se2 kernels
        # With shape: [orientations_nb, kernelSizeH, kernelSizeW, orientations_nb,
        # channelsIN, channelsOUT]
        # Kernel dimensions
        kernelSizeH, kernelSizeW, orientations_nb, channelsIN, channelsOUT = map(
            int, self.w.shape)
        kernel_stack = rotate_gconv_kernels(self.w, self.periodicity,
                                            self.disk_mask)
        # print("SE2N-SE2N ROTATED KERNEL SET SHAPE:",
        #       kernel_stack.get_shape())  # Debug
        kernels_as_if_2D = tf.transpose(kernel_stack, [1, 2, 3, 4, 0, 5])
        kernels_as_if_2D = tf.reshape(
            kernels_as_if_2D, [kernelSizeH, kernelSizeW,
                               orientations_nb * channelsIN, orientations_nb * channelsOUT])
        return kernels_as_if_2D, kernel_stack

    def call(self, inputs, **kwargs):
        # The group convolution layer
        input_tensor_as_if_2D = tf.reshape(inputs,
                                           [tf.shape(inputs)[0],
                                            int(inputs.shape[1]),
                                            int(inputs.shape[2]),
                                            int(inputs.shape[3]*inputs.shape[4])])

        kernels_as_if_2D, _ = self.rotate_kernel()

        output = tf.nn.conv2d(
            input=input_tensor_as_if_2D,
            filters=kernels_as_if_2D,
            strides=self.strides,
            padding=self.padding)

        output = tf.reshape(
            output, [tf.shape(output)[0], int(output.shape[1]), int(output.shape[2]),
                     self.n_theta, self.filters])
        # print("OUTPUT SE2N ACTIVATIONS SHAPE:", output.get_shape())  # Debug
        if self.use_bias:
            output += self.b
            
        return self.activation(output)


class ConcatOrientation(tf.keras.layers.Layer):

    def build(self, input_shape):
        dx = input_shape[1]
        dy = input_shape[2]
        n_theta = input_shape[3]
        channels = input_shape[4]
        self.reshape = layers.Reshape((dx, dy, n_theta*channels))

    def call(self, inputs, **kwargs):

        return self.reshape(inputs)


class SpatialMaxpool(tf.keras.layers.Layer):
    def __init__(self, n_theta=12, **kwargs):
        super().__init__(**kwargs)
        self.n_theta = n_theta

    def call(self, inputs, **kwargs):
        return spatial_max_pool(inputs, self.n_theta)
