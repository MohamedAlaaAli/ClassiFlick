# class Segmentation:
#     def __init__(self, image):
#         self.image = image

#     def K_means(self):
#         pass

#     def MeanShift(self):
#         pass

#     def agglomerative_clustering(self):
#         pass

#     def region_growing(self):
#         pass


import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from collections import deque

def calculate_distance(cluster1, cluster2):
    # Calculate the Euclidean distance between two clusters in Luv color space
    return np.linalg.norm(cluster1 - cluster2)

def merge_clusters(cluster1, cluster2):
    # Merge two clusters by taking the mean of their values
    return (cluster1 + cluster2) / 2

def calculate_local_min_distances(clusters, start_idx, end_idx):
    # Calculate local minimum distances between clusters within a specified range
    local_min_distances = []
    for i in range(start_idx, end_idx):
        min_dist = float('inf')
        min_dist_idx = None
        for j in range(i+1, len(clusters)):
            dist = calculate_distance(clusters[i], clusters[j])
            if dist < min_dist:
                min_dist = dist
                min_dist_idx = j
        local_min_distances.append((i, min_dist_idx, min_dist))
    return local_min_distances

def agglomerative_segmentation(image, num_clusters, num_threads=4):
    # Convert image to Luv color space
    image_luv = cv2.cvtColor(image, cv2.COLOR_BGR2Luv)
    
    # Initialize clusters
    clusters = image_luv.reshape(-1, 3)
    
    # Multithreaded calculation of local minimum distances
    local_min_distances = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        chunk_size = len(clusters) // num_threads
        for i in range(0, len(clusters), chunk_size):
            end_idx = min(i + chunk_size, len(clusters))
            futures.append(executor.submit(calculate_local_min_distances, clusters, i, end_idx))
        for future in futures:
            local_min_distances.extend(future.result())

    # Perform agglomerative clustering
    while len(clusters) > num_clusters:
        # Find the global minimum distance from local minimum distances
        min_dist = float('inf')
        min_dist_indices = None
        for dist_info in local_min_distances:
            if dist_info[2] < min_dist:
                min_dist = dist_info[2]
                min_dist_indices = dist_info
        
        # Merge the two clusters with the minimum distance
        merged_cluster = merge_clusters(clusters[min_dist_indices[0]], clusters[min_dist_indices[1]])
        clusters[min_dist_indices[0]] = merged_cluster
        del clusters[min_dist_indices[1]]
        
        # Update local minimum distances
        new_local_min_distances = []
        for dist_info in local_min_distances:
            i, j, dist = dist_info
            if j != min_dist_indices[1]:
                if j > min_dist_indices[1]:
                    j -= 1
                new_local_min_distances.append((i, j, calculate_distance(clusters[i], clusters[j])))
        local_min_distances = new_local_min_distances

    # Reshape the clustered image
    segmented_image_luv = np.array(clusters, dtype=np.uint8).reshape(image_luv.shape)

    # Convert segmented image back to BGR color space
    segmented_image_bgr = cv2.cvtColor(segmented_image_luv, cv2.COLOR_Luv2BGR)

    return segmented_image_bgr

def region_growing(image, seeds, threshold, window_size):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    visited = np.zeros_like(gray_image, dtype=bool)

    segmented = np.zeros_like(image)
    half_window = window_size // 2

    for seed in seeds:
        x_seed, y_seed = seed[0], seed[1]
        if 0 <= x_seed < gray_image.shape[0] and 0 <= y_seed < gray_image.shape[1]:
            mean_value = gray_image[x_seed, y_seed]
            number_of_pixels_in_region = 1  # Initialize to 1 for the seed pixel
            queue = deque([(x_seed, y_seed)])

            while len(queue):
                current_x, current_y = queue.popleft()
                if (0 <= current_x < gray_image.shape[0] and 0 <= current_y < gray_image.shape[1]) and not visited[current_x, current_y]:
                    visited[current_x, current_y] = True

                    if abs(gray_image[current_x, current_y] - mean_value) < threshold:
                        # Update mean value
                        mean_value = ((mean_value * number_of_pixels_in_region + gray_image[current_x, current_y]) /
                                      (number_of_pixels_in_region + 1))
                        number_of_pixels_in_region += 1

                        # Mark the pixel as part of the region
                        segmented[current_x, current_y] = image[current_x, current_y]

                        # Add neighboring pixels to the queue
                        for x in range(- half_window, half_window + 1):
                            for y in range(- half_window, half_window + 1):
                                if 0 <= current_x + x < gray_image.shape[0] and 0 <= current_y + y < gray_image.shape[1]:
                                    queue.append((current_x + x, current_y + y))

    visualize_regions(image, segmented)



def visualize_regions(image, segmented_image):
    contours, _ = cv2.findContours(
        cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # Draw contours on input image
    output_image = image.copy()
    cv2.drawContours(output_image, contours, -1, (255, 0, 0), 2)

    # Display the output image
    cv2.imshow('Segmented Image', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Load the image
image = cv2.imread('Brain2.pgm')
region_growing(image, [(500, 400)], 50, 3)
# Perform agglomerative segmentation with multithreading
# num_clusters = 5
# segmented_image = agglomerative_segmentation(image, num_clusters)
#
# # Display original and segmented images
# cv2.imshow('Original Image', image)
# cv2.imshow('Segmented Image', segmented_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
