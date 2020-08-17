import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from tensor_utils import create_anchor_boxes_tensor, create_offsets, SMALL_ANCHOR_BOXES, MEDIUM_ANCHOR_BOXES, LARGE_ANCHOR_BOXES
import math

IMAGE_WIDTH = 416
LARGE_ANCHOR_BOXES_TENSOR = create_anchor_boxes_tensor(LARGE_ANCHOR_BOXES, 13)
MEDIUM_ANCHOR_BOXES_TENSOR = create_anchor_boxes_tensor(MEDIUM_ANCHOR_BOXES, 26)
SMALL_ANCHOR_BOXES_TENSOR = create_anchor_boxes_tensor(SMALL_ANCHOR_BOXES, 52)

LARGE_OFFSETS = create_offsets(13)
MEDIUM_OFFSETS = create_offsets(26)
SMALL_OFFSETS = create_offsets(52)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def get_boxes_per_array(y_pred):
    n_grid = y_pred.shape[2]
    anchor_boxes_tensor = []
    offsets = []
    if (n_grid == 13):
        offsets = LARGE_OFFSETS
        anchor_boxes_tensor = LARGE_ANCHOR_BOXES_TENSOR
    elif (n_grid == 26):
        offsets = MEDIUM_OFFSETS
        anchor_boxes_tensor = MEDIUM_ANCHOR_BOXES_TENSOR
    else:
        offsets = SMALL_OFFSETS
        anchor_boxes_tensor = SMALL_ANCHOR_BOXES_TENSOR

    pred_t_xy = tf.cast(y_pred[..., 1:3], tf.float32)
    pred_t_wh = tf.cast(y_pred[..., 3:5], tf.float32)

    # bx, by = sigma(tx, ty) + cx, cy
    pred_b_xy = tf.cast(tf.nn.sigmoid(pred_t_xy), tf.float32) + tf.cast(offsets[0, ...], tf.float32)

    # bw, bh = (pw, ph) * exp(tw, th)
    # print('Calculating wh')
    pred_b_wh = anchor_boxes_tensor[0, ...] * tf.cast(tf.math.exp(pred_t_wh), tf.float32)
    predicted_objecteness = tf.cast(tf.nn.sigmoid(y_pred[..., 0]), tf.float32)
    # print(pred_b_wh[0, :, :, 0])

    xy = pred_b_xy.numpy() * IMAGE_WIDTH / n_grid
    wh = pred_b_wh.numpy() * IMAGE_WIDTH

    left_top_corner = xy - wh / 2
    right_bottom_corner = xy + wh / 2

    # print(right_bottom_corner[0, :, :, 1])

    # print()
    boxes = []

    for i in range(0, len(LARGE_ANCHOR_BOXES)):
        for j in range(0, n_grid):
            for k in range(0, n_grid):
                predicted_objectness = predicted_objecteness[i, j, k]
                confidence_threshhold = 0.6
                if (predicted_objectness > confidence_threshhold):
                    # print(x_offset, y_offset)
                    # print(left_top_corner[i,j,k,0])
                    x0 = int(left_top_corner[i, j, k, 0])
                    y0 = int(left_top_corner[i, j, k, 1])
                    x1 = int(right_bottom_corner[i, j, k, 0])
                    y1 = int(right_bottom_corner[i, j, k, 1])
                    boxes.append([x0, y0, x1, y1])
    return boxes


def draw_pred_image(image, small_predicted, medium_predicted, large_predicted, NMS=False):
    image = Image.fromarray(np.uint8((image) * 255))
    pred_d = ImageDraw.Draw(image)

    boxes = get_boxes_per_array(small_predicted) + get_boxes_per_array(medium_predicted) + get_boxes_per_array(
        large_predicted)

    boxes = np.array(boxes)
    if (len(boxes) > 0 and NMS):
        nms_threshhold = 0.4  # @param {type:"slider", min:0, max:1.0, step:0.05}
        boxes = non_max_suppression_fast(np.array(boxes), nms_threshhold)

    for i in range(boxes.shape[0]):
        pred_d.rectangle([(boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3])])
    return image


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")
