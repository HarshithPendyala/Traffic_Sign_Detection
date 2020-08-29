import numpy as np
import cv2


def preprocessing(img):
    im = img.copy()

    # Normalize image
    normalised = np.zeros((im.shape[0], im.shape[1]), np.uint8)
    im_cast = im.astype(int)
    im_sum = np.sum(im_cast, axis=2, keepdims=True)
    normalised = (1.0 * im / (1.0 * im_sum + 1e-7)) * 255.0
    kernel = np.ones((3, 3), np.uint8)



    # Generating red mask
    normalised_red = normalised[:, :, 2].astype(np.uint8)
    norm_hist_red = cv2.calcHist([normalised_red], [0], None, [256], [0, 256])
    norm_hist_rev_red = norm_hist_red[::- 1]
    sum_pix_red = np.sum(norm_hist_rev_red)
    cum_sum_pix_red = np.cumsum(norm_hist_rev_red)
    allowed_pixs_red = sum_pix_red * 0.002
    thresh_red = 255 - np.argmax(norm_hist_rev_red > allowed_pixs_red)
    re, thresholded_red = cv2.threshold(normalised_red, thresh_red, 255, cv2.THRESH_BINARY)
    erosion_red = cv2.erode(thresholded_red, kernel, iterations=1)
    dilate_red = cv2.dilate(erosion_red, kernel, iterations=0)
    median_red = cv2.medianBlur(dilate_red, 3)

    # Generating blue mask
    normalised_blue = normalised[:, :, 0].astype(np.uint8)
    norm_hist_blue = cv2.calcHist([normalised_blue], [0], None, [256], [0, 256])
    norm_hist_rev_blue = norm_hist_blue[::- 1]
    sum_pix_blue = np.sum(norm_hist_rev_blue)
    cum_sum_pix_blue = np.cumsum(norm_hist_rev_blue)
    allowed_pixs_blue = sum_pix_blue * 0.004
    thresh_blue = 255 - np.argmax(norm_hist_rev_blue > allowed_pixs_blue)
    bl, thresholded_blue = cv2.threshold(normalised_blue, thresh_blue, 255, cv2.THRESH_BINARY)
    # erosion_blue = cv2.erode(thresholded_blue, kernel, iterations=1)
    # dilate_blue = cv2.dilate(erosion_blue, kernel, iterations=1)
    median_blue = cv2.medianBlur(thresholded_blue, 3)



    return median_red, median_blue
