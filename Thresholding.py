import cv2
import numpy as np

class Thresholding:
    def __init__(self, image):
        self.image = image

    def __optimal_threshold(
        self, hist: np.ndarray, initial_threshold: int, max_iter: int = 100, min_diff: float = 1e-5) -> int:
        """
        Perform an optimal threshold algorithm on a histogram.

        Parameters:
            hist (np.ndarray[int]): The input histogram.
            initial_threshold (int): The initial threshold value.
            max_iter (int, optional): The maximum number of iterations. Defaults to 100.
            min_diff (float, optional): The tolerance for convergence. Defaults to 1e-5.

        Returns:
            int: The optimal threshold value.
        """
        # Normalize the histogram
        hist_norm = hist / hist.sum()

        # Initialize threshold
        t = initial_threshold

        for _ in range(max_iter):
            # Cut the distribution into two parts
            h1 = hist_norm[:t]
            h2 = hist_norm[t:]

            # Compute centroids, mean of the two parts
            m1 = (np.arange(t, dtype=np.float64) * h1).sum() / (h1.sum() + 1e-5)
            m2 = (np.arange(t, 256, dtype=np.float64) * h2).sum() / (h2.sum() + 1e-5)

            # Compute new threshold
            t_new = int(round((m1 + m2) / 2))

            # Check convergence
            if abs(t_new - t) < min_diff:
                break

            t = t_new

        return t

    def optimal_threshold(
            self, initial_threshold: int = 125, max_iter: int = 100, min_diff: float = 1e-5) -> tuple[int, np.ndarray]:
        """
        Calculates the optimal threshold value for image segmentation using the optimal thresholding algorithm.

        Parameters:
            initial_threshold (int): The initial threshold value. Default is 125.
            max_iter (int): The maximum number of iterations for the algorithm. Default is 100.
            min_diff (float): The tolerance for convergence. Default is 1e-5.

        Returns:
            tuple: A tuple containing the optimal threshold value (int) and the thresholded image (numpy array).
        """
        # Convert the image to grayscale if it's not already
        if self.image.ndim == 3:
            gray: np.ndarray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image

        # Compute the histogram
        hist: np.ndarray = cv2.calcHist([gray], [0], None, [256], [0,256])

        # Reshape the histogram to 1D array
        hist = hist.flatten()

        optimal_t: int = self.__optimal_threshold(hist, initial_threshold, max_iter, min_diff)

        _, thresholded_image = cv2.threshold(gray, optimal_t, 255, cv2.THRESH_BINARY)

        return optimal_t, thresholded_image


    def otsu_threshold(self):
        pass

    def spectral_threshold(self):
        pass


def main():
    # Example usage:
    # Read the input image
    image = cv2.imread('images\Screenshot 2024-04-21 232347.png')

    # Create an instance of the Thresholding class
    thresholding = Thresholding(image)

    # Calculate the optimal threshold and get the thresholded image
    optimal_t, thresholded_image = thresholding.optimal_threshold(initial_threshold=10)
    print("Optimal threshold:", optimal_t)

    # Display the thresholded image
    cv2.imshow(f'Thresholded Image with value{optimal_t}', thresholded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()




