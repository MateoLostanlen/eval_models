import numpy as np


def xywh2xyxy(x):
    """Function to convert bounding box format from center to top-left corner"""
    y = np.zeros_like(x)
    y[0] = x[0] - x[2] / 2  # x_min
    y[1] = x[1] - x[3] / 2  # y_min
    y[2] = x[0] + x[2] / 2  # x_max
    y[3] = x[1] + x[3] / 2  # y_max
    return y


def box_iou(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (np.ndarray): A numpy array of shape (N, 4) representing N bounding boxes.
        box2 (np.ndarray): A numpy array of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.ndarray): An NxM numpy array containing the pairwise IoU values for every element in box1 and box2.
    """

    # Ensure box1 and box2 are in the shape (N, 4) even if N is 1
    if box1.ndim == 1:
        box1 = box1.reshape(1, 4)
    if box2.ndim == 1:
        box2 = box2.reshape(1, 4)

    (a1, a2), (b1, b2) = np.split(box1, 2, 1), np.split(box2, 2, 1)
    inter = (
        (np.minimum(a2, b2[:, None, :]) - np.maximum(a1, b1[:, None, :]))
        .clip(0)
        .prod(2)
    )

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(1) + (b2 - b1).prod(1)[:, None] - inter + eps)
