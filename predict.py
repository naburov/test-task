import tensorflow as tf
from model import yolov3
from image_utils import draw_pred_image
import numpy as np

def predict(open_path, save_path, confidence_thresh):
    image = load_and_preprocess_image(open_path)
    print('Image opened')
    model = yolov3()
    model.load_weights('./weights/yolov3')
    print('Weights loaded')
    small_array, medium_array, large_array = model(image[tf.newaxis, ...])
    print('Predict received')
    image = draw_pred_image(image, small_array[0, ...], medium_array[0, ...], large_array[0, ...], confidence_thresh)
    image.save(save_path, "JPEG")
    print('Saved')
    return


def load_and_preprocess_image(path):
    image_file = tf.io.read_file(path)
    image = tf.image.decode_image(image_file, channels=3)
    image = tf.image.resize(image, [416, 416])
    image /= 255.0
    return image