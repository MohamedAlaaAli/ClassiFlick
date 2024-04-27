import cv2
import numpy as np

class SpectralThresholding:
    def __init__(self, image):
        self.image = image


    def global_threshold(self):
        gray_image = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)

        hist = np.zeros(256)
        for i in range(gray_image.shape[0]):
            for j in range(gray_image.shape[1]):
                hist[gray_image[i, j]] += 1

        hist_cdf = np.cumsum(hist)
        total_pixels = hist_cdf[-1]
        threshold_low = np.argmax(hist_cdf >= total_pixels / 3)
        threshold_high= np.argmax(hist_cdf >= total_pixels * 2 / 3)

        global_image  = np.zeros_like(gray_image)
        global_image [gray_image <= threshold_low] = 0
        global_image [(gray_image > threshold_low) & (gray_image <= threshold_high)] = 127
        global_image [gray_image > threshold_high] = 255
        

        return global_image

    def local_threshold(self, block_size=20, bandwidth=5):

        gray_image = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)

        local_image = np.zeros_like(gray_image)

        pad = block_size // 2
        
        for i in range(pad, gray_image.shape[0] - pad):
            for j in range(pad, gray_image.shape[1] - pad):
                block = gray_image[i - pad:i + pad + 1, j - pad:j + pad + 1]
                block_mean = np.mean(block)
                threshold_low = block_mean - bandwidth
                threshold_high = block_mean + bandwidth
                if gray_image[i, j] <= threshold_low:
                    local_image[i, j] = 0
                elif gray_image[i, j] > threshold_low and gray_image[i, j] <= threshold_high:
                    local_image[i, j] = 127
                else:
                    local_image[i, j] = 255

        return local_image


if __name__ == "__main__":

    image = cv2.imread("images/shore.jpg")
    spectral_thresholding = SpectralThresholding(image)

    global_image = spectral_thresholding.global_threshold()

    local_image = spectral_thresholding.local_threshold()
    cv2.imshow("Global Image", global_image)
    cv2.imshow("Local Image", local_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
