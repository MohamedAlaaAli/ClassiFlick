import numpy as np
import cv2

class MeanShiftSegmentation:
    def __init__(self, image):
        self.image = image

    def segment_image(self):
        image_height, image_width, _ = self.image.shape
        segmented_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        color_list = self.image.reshape(-1, 3)
        position_list = np.indices((image_height, image_width)).reshape(2, -1).T
        color_and_position_list = np.hstack((color_list, position_list))

        # Random mean
        current_mean = np.zeros(5)
        current_mean = color_and_position_list[np.random.randint(0, color_and_position_list.shape[0])]

        while color_and_position_list.shape[0] > 0:
                
            distances = self.get_distances(color_and_position_list[:, :3], current_mean)
            distances = np.where(distances < 100)[0] # indices

            mean_color = np.mean(color_and_position_list[distances, :3], axis=0)
            mean_position = np.mean(color_and_position_list[distances, 3:], axis=0)
            color_distance_to_mean = np.sqrt(np.sum((mean_color - current_mean[:3]) ** 2))

            position_distance_to_mean = np.sqrt(np.sum((mean_position - current_mean[3:]) ** 2))

            total_distance = color_distance_to_mean + position_distance_to_mean

            if total_distance < 200: # Threshold
                new_color = np.zeros(3)
                new_color = mean_color
                segmented_image[
                    color_and_position_list[distances, 3],
                    color_and_position_list[distances, 4],
                ] = new_color
                color_and_position_list = np.delete(color_and_position_list, distances, axis=0)

                # New random mean
                if color_and_position_list.shape[0] > 0:
                    current_mean = color_and_position_list[np.random.randint(0, color_and_position_list.shape[0])]

            else:
                current_mean[:3] = mean_color
                current_mean[3:] = mean_position

        return segmented_image


    def get_distances(self, color_and_position_list, current_mean):
        distances = np.zeros(color_and_position_list.shape[0])
        for i in range(len(color_and_position_list) - 1):
            distance = 0
            for j in range(3):
                distance += (current_mean[j] - color_and_position_list[i][j]) ** 2
            distances[i] = distance ** 0.5
        return distances


if __name__ == "__main__":

    image = cv2.imread('images/shore.jpg')
    mean_shift = MeanShiftSegmentation(image)

    shifted_image = mean_shift.segment_image()
    
    cv2.imshow('Original Image', image)
    cv2.imshow('Shifted Image', shifted_image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
