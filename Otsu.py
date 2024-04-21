import numpy as np
import cv2

def otsu(img):
    """
    Applies Otsu's thresholding algorithm to find the optimal threshold.

    Args:
        img (np.ndarray): Input grayscale image.

    Returns:
        int: The optimal threshold value.
    """
    thresholds = []
    for t in range(img.min() + 1, img.max()):
        below_threshold = img[img < t]
        above_threshold = img[img >= t]
        Wb = len(below_threshold) / (img.shape[0] * img.shape[1])
        Wa = len(above_threshold) / (img.shape[0] * img.shape[1])
        var_b = np.var(below_threshold)
        var_a = np.var(above_threshold)
        thresholds.append(Wb * var_b + var_a * Wa)
    try:
        min_threshold = min(thresholds)
        optimal_threshold = thresholds.index(min_threshold)
    except ValueError:
        optimal_threshold = 0
    return optimal_threshold

def apply_otso(img):
    """
    Applies Otsu's thresholding algorithm to find the optimal threshold.

    Args:
        img (np.ndarray): Input grayscale image.

    Returns:
        np.ndarray: The thresholded image.
    """
    if img.ndim != 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    threshold = otsu(img)
    print("aaa")
    thresholded_image = np.where(img >= img.min() + threshold, 255, 0).astype(np.uint8)
    
    return thresholded_image

image = cv2.imread('images/Image_processing_pre_otsus_algorithm.jpg')
thresholded_image = apply_otso(image)
cv2.imshow('Thresholded Image', thresholded_image)
cv2.waitKey(0)