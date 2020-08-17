import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization, LeakyReLU


class DarkNetConvBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides):
        super(DarkNetConvBlock, self).__init__(name='')

        self.conv2a = tf.keras.layers.Conv2D(filters, (1, 1), strides=strides)
        self.bn2a = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False, mask=None):
        x = self.conv2a(inputs)
        x = self.bn2a(x, training=training)

        return tf.nn.leaky_relu(x)

class DarkNetResidualBlock(tf.keras.Model):
  def __init__(self, kernel_size, filters):
    super(DarkNetResidualBlock, self).__init__(name='')
    filters1, filters2 = filters

    self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
    self.bn2a = tf.keras.layers.BatchNormalization()

    self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
    self.bn2b = tf.keras.layers.BatchNormalization()

  def call(self, input_tensor, training=False, mask=None):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.leaky_relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.leaky_relu(x)

    x += input_tensor
    return tf.nn.leaky_relu(x)


def DarkNet53(inputs):
    x = DarkNetConvBlock(32, (3, 3), strides=(1, 1))(inputs)
    x = DarkNetConvBlock(64, (3, 3), strides=(2, 2))(x)

    x = DarkNetResidualBlock((3, 3), [32, 64])(x)

    x = DarkNetConvBlock(128, (3, 3), strides=(2, 2))(x)

    x = DarkNetResidualBlock((3, 3), [64, 128])(x)
    x = DarkNetResidualBlock((3, 3), [64, 128])(x)

    x = DarkNetConvBlock(256, (3, 3), strides=(2, 2))(x)

    x = DarkNetResidualBlock((3, 3), [128, 256])(x)
    x = DarkNetResidualBlock((3, 3), [128, 256])(x)
    # x = DarkNetResidualBlock((3,3), [128, 256])(x)
    # x = DarkNetResidualBlock((3,3), [128, 256])(x)
    x = DarkNetResidualBlock((3, 3), [128, 256])(x)
    x = DarkNetResidualBlock((3, 3), [128, 256])(x)
    x = DarkNetResidualBlock((3, 3), [128, 256])(x)
    block_large_out = DarkNetResidualBlock((3, 3), [128, 256])(x)

    x = DarkNetConvBlock(512, (3, 3), strides=(2, 2))(block_large_out)

    x = DarkNetResidualBlock((3, 3), [256, 512])(x)
    x = DarkNetResidualBlock((3, 3), [256, 512])(x)
    x = DarkNetResidualBlock((3, 3), [256, 512])(x)
    # x = DarkNetResidualBlock((3,3), [256, 512])(x)
    # x = DarkNetResidualBlock((3,3), [256, 512])(x)
    x = DarkNetResidualBlock((3, 3), [256, 512])(x)
    x = DarkNetResidualBlock((3, 3), [256, 512])(x)
    block_medium_out = DarkNetResidualBlock((3, 3), [256, 512])(x)

    x = DarkNetConvBlock(1024, (3, 3), strides=(2, 2))(block_medium_out)

    x = DarkNetResidualBlock((3, 3), [512, 1024])(x)
    x = DarkNetResidualBlock((3, 3), [512, 1024])(x)
    x = DarkNetResidualBlock((3, 3), [512, 1024])(x)
    block_small_out = DarkNetResidualBlock((3, 3), [512, 1024])(x)
    return tf.keras.Model(inputs=inputs, outputs=[block_small_out, block_medium_out, block_large_out])


def yolov3():
    inputs = tf.keras.layers.Input(shape=(416, 416, 3), batch_size=4)
    darknet = DarkNet53(inputs)
    small, medium, large = darknet(inputs)

    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same')(small)
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same')(x)
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(x)
    to_upsample = Conv2D(512, (1, 1), strides=(1, 1), padding='same')(x)
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(3 * 5, (1, 1), strides=(1, 1), padding='same')(x)
    large_scale_output = tf.keras.layers.Reshape((3, 13, 13, 5))(x)

    x = tf.keras.layers.UpSampling2D()(to_upsample)
    x = tf.keras.layers.Concatenate()([medium, x])
    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same')(x)
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same')(x)
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(x)
    to_upsample = Conv2D(512, (1, 1), strides=(1, 1), padding='same')(x)
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(3 * 5, (1, 1), strides=(1, 1), padding='same')(x)
    medium_scale_output = tf.keras.layers.Reshape((3, 26, 26, 5))(x)

    x = tf.keras.layers.UpSampling2D()(to_upsample)
    x = tf.keras.layers.Concatenate()([large, x])
    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same')(x)
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same')(x)
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(3 * 5, (1, 1), strides=(1, 1), padding='same')(x)
    small_scale_output = tf.keras.layers.Reshape((3, 52, 52, 5))(x)
    return tf.keras.Model(inputs, outputs=[small_scale_output, medium_scale_output, large_scale_output])
